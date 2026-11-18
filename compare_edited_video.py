#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_edited_video.py — 手动剪辑 vs 自动剪辑：字幕 + 结构 + 文字风格全面对比

流程（尽量复用现有代码，不重复实现）:
    1. 字幕：对两个文件夹里的每个视频调用 sensevoice_srt.py
       （抽音频 -> VAD 断句 -> SenseVoice 识别 -> 大模型纠错 -> 主播声纹标注），
       SRT / 断点 JSONL 落到缓存目录，已有就跳过（--force-asr 重跑）。
       解析字幕直接用 edit_with_caption.read_srt（连「（主播）」标记一起读回来）。
    2. 画面切换：ffmpeg scene 打分，得到「镜头切换密度」（缓存成 json）。
    3. 指标：说话密度、主播说话比例、语速、段落切分（停顿 > SEG_GAP_SEC 算新段）
       -> 段数 / 段长中位数 / 最长段 / 每分钟段数；文字风格（问句率、感叹率、
       语气词、口头禅、互动词、用字多样性）；jieba 词频 + 两组 log-odds 特征词。
    4. 大模型：把段落归类成内容类型（剧情 / 战斗 / 读弹幕 / 吐槽 …），
       再写总体对比 + 每个视频的特点（--no-llm 关闭，只留规则版总结）。
    5. 出图 + 报告：compare\\report\\ 里的 png 图表、metrics.json / metrics.csv、
       summary.md（文字总结）。

用法:
    python compare_edited_video.py                        # 全量跑（慢，字幕要识别很久）
    python compare_edited_video.py --limit-minutes 3      # 试跑：每个视频只识别前 3 分钟
    python compare_edited_video.py --only day3            # 只跑文件名含 day3 的视频
    python compare_edited_video.py --force-asr            # 已有字幕也重新识别
    python compare_edited_video.py --no-cuts --no-llm     # 不出画面切换 / 不调大模型

注意: 字幕纠错（--no-correct 关闭）和内容归类、总结都要 DEEPSEEK_API_KEY。
      没有 key 时自动降级：字幕不纠错、报告只出规则版文字总结。
"""

import os
import re
import sys
import json
import math
import time
import shutil
import argparse
import subprocess
import tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 复用现有代码：字幕解析 / 文件枚举 / ffprobe / 大模型调用
import edit_with_caption as ewc
import sensevoice_srt as sv

# ---------------------------------------------------------------------------
# 输入 / 输出
# ---------------------------------------------------------------------------
MANUAL_DIR = os.path.join(HERE, "compare", "manual")   # 手动剪辑
AUTO_DIR = os.path.join(HERE, "compare", "auto")       # 自动剪辑
REPORT_DIR = os.path.join(HERE, "compare", "report")   # 报告输出目录

GROUP_LABEL = {"manual": "手动剪辑", "auto": "自动剪辑"}
GROUP_COLOR = {"manual": "#4C72B0", "auto": "#DD8452"}
STREAMER_COLOR = "#D1495B"
OTHER_COLOR = "#4C72B0"

# ---------------------------------------------------------------------------
# 第 1 步：字幕（全部交给 sensevoice_srt.py）
# ---------------------------------------------------------------------------
ASR_DEVICE = "auto"                 # auto / cuda:0 / cpu
ASR_LANGUAGE = "zh"
ASR_CORRECT = True                  # 字幕纠错（需要 DEEPSEEK_API_KEY）
ASR_MARK_STREAMER = True            # 字幕里标注（主播）
ASR_STREAMER_REF = r"video_source\sample voice\文七3.2视频.mp4"   # 空 = 自动注册
ASR_STREAMER_THRESHOLD = 0.0        # 0 = 用 sensevoice_srt.py 的默认阈值
ASR_REF_SEC = 0.0                   # 0 = 用 sensevoice_srt.py 的默认值
ASR_MAX_HOURS = 0.0                 # 0 = 用 sensevoice_srt.py 的默认上限（9 小时）

# ---------------------------------------------------------------------------
# 指标参数
# ---------------------------------------------------------------------------
SEG_GAP_SEC = 3.0                   # 停顿超过这个秒数就切新的一段
SEG_MIN_SEC = 8.0                   # 短于这个秒数的碎段并进相邻段
SEG_MAX_SEC = 240.0                 # 单段最长（更长的在最大停顿处再切）

SCENE_DETECT = True                 # 是否做画面切换检测
SCENE_THRESHOLD = 0.30              # ffmpeg scene 阈值（越大越只留硬切）
SCENE_FPS = 5                       # 采样帧率（5 帧/秒足够抓切点）

# ---------------------------------------------------------------------------
# 大模型
# ---------------------------------------------------------------------------
LLM_TOPIC_BATCH = 20                # 每次请求归类多少段
LLM_TOPIC_MIN_SEC = 15.0            # 太短的段不归类（信息量不够）
LLM_MAX_SEGMENTS_PER_VIDEO = 80     # 每个视频最多归类多少段（均匀抽样）
LLM_SUMMARY = True                  # 是否让大模型写总体总结 / 每片特点

# 内容类型（大模型只能从这里面选）
TOPICS = ["剧情/主线", "战斗/操作", "抽卡/养成", "读弹幕互动", "杂谈闲聊",
          "吐槽/玩梗", "攻略/讲解", "情绪高光", "其他"]

# 类型定义要和 edit_with_caption.py 保持一致，两边算出来的配比才能直接对比
TOPIC_GUIDE = """- 剧情/主线：游戏剧情推进、主线对话、关键演出（主播在旁边解说/反应）
- 战斗/操作：打怪、操作、走位、Boss 战
- 抽卡/养成：抽卡、角色培养、开荒、配队
- 读弹幕互动：念弹幕、回答观众问题、和观众互动
- 杂谈闲聊：主播随口聊日常、感想、和游戏无关的话题
- 吐槽/玩梗：对游戏内容/事件的吐槽、玩梗、整活、情绪化评价（有明确吐槽对象）
- 攻略/讲解：讲机制、讲思路、教学
- 情绪高光：纯情绪爆发（惊讶、爆笑、破防），没有明确吐槽对象和内容
- 其他：以上都不像"""

MODAL_CHARS = set("啊呀呢吧嘛哦噢诶唉嗯哈嘿哟啦喔哇咯呗喽")

# 口头禅 / 高频口语（多字为主，避免和普通词混淆）
CATCHPHRASES = list(dict.fromkeys([
    "就是", "然后", "这个", "那个", "不是", "其实", "反正", "真的", "感觉", "有点",
    "我靠", "卧槽", "哈哈", "笑死", "离谱", "抽象", "好家伙", "对吧", "是吧",
    "好吧", "不对", "完了", "绝了", "懂了", "啊这", "绷不住", "无语", "哎哟",
    "确实", "直接", "居然", "竟然", "突然", "为什么", "怎么",
]))

# 和观众互动的词（自动剪辑的提示词里要求「避免变成读弹幕」，可以验证一下）
INTERACT_WORDS = ["弹幕", "观众", "兄弟", "家人", "sc", "引流", "送礼", "打赏"]

STOPWORDS = set("""
我们 你们 他们 她们 它们 咱们 自己 大家 这个 那个 这种 那种 这样 那样 这里 那里
什么 怎么 为什么 哪里 哪个 多少 几个 一下 一点 一些 一个 一直 一样 一起 一定
就是 然后 但是 因为 所以 而且 不过 只是 还有 或者 如果 虽然 已经 还是 也是 不是
可以 没有 知道 觉得 应该 时候 东西 地方 问题 事情 意思 感觉 现在 刚才 之后 之前
比较 非常 特别 真的 其实 反正 有点 可能 应该 需要 这样 那样 的话 起来 出来 下来
上去 下去 进来 回来 过去 过来 知道 现在 一样 怎么 什么 为什么 哪里 自己 我们
""".split())

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_WORD_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_ALNUM_RE = re.compile(r"[0-9A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def fmt_hms(seconds):
    x = int(max(round(float(seconds)), 0))
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def fmt_min(seconds):
    return f"{float(seconds) / 60.0:.1f}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def content_len(text):
    """只数实字（汉字 / 数字 / 字母），不算标点，用来算语速。"""
    return len(_CJK_RE.findall(text or "")) + len(_ALNUM_RE.findall(text or ""))


def is_streamer(entry):
    return "主播" in (entry.get("speaker") or "")


def short_name(name):
    """图表用短名字：去掉「文七」「_edited」「录播」「full」这类噪音。"""
    s = os.path.splitext(os.path.basename(name))[0]
    for a in ("文七 ", "文七", "_edited", "录播", " full", "Full"):
        s = s.replace(a, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:16] or os.path.basename(name)


def variant_suffix(limit_minutes):
    """试跑（--limit-minutes）时给缓存文件名加后缀，避免污染正式结果。"""
    return "" if not limit_minutes or limit_minutes <= 0 else f"_head{limit_minutes:g}min"


def pct(a, b):
    try:
        return 100.0 * float(a) / float(b)
    except ZeroDivisionError:
        return 0.0


def rel_diff(a, b):
    """a 相对 b 的变化百分比，用于文字总结。"""
    if not b:
        return "n/a"
    return f"{(a - b) / b * 100:+.0f}%"


# ---------------------------------------------------------------------------
# 第 1 步：字幕（子进程调用 sensevoice_srt.py，完全复用现有流程）
# ---------------------------------------------------------------------------
def cache_dir(root, kind, group):
    return ensure_dir(os.path.join(root, "cache", kind, group))


def asr_files(root, group, video, suffix=""):
    stem = os.path.splitext(os.path.basename(video))[0] + suffix
    d = cache_dir(root, "asr", group)
    return os.path.join(d, stem + ".srt"), os.path.join(d, stem + ".asr.jsonl")


def build_asr_cmd(video, srt, jsonl, args):
    cmd = [sys.executable, os.path.join(HERE, "sensevoice_srt.py"), video,
           "-o", srt, "--jsonl", jsonl,
           "--device", args.device, "--language", args.language]
    if os.path.isfile(jsonl):
        cmd.append("--resume")          # 有断点就接着跑
    if args.limit_minutes > 0:
        cmd += ["--limit-minutes", f"{args.limit_minutes:g}"]
    if not args.correct:
        cmd += ["--no-correct"]
    if not args.mark_streamer:
        cmd += ["--no-streamer-mark"]
    else:
        if args.streamer_ref:
            ref = args.streamer_ref
            if not os.path.isabs(ref):
                ref = os.path.join(HERE, ref)
            cmd += ["--streamer-ref", ref]
        if args.streamer_threshold > 0:
            cmd += ["--streamer-threshold", f"{args.streamer_threshold:g}"]
        if args.ref_sec > 0:
            cmd += ["--ref-sec", f"{args.ref_sec:g}"]
    if ASR_MAX_HOURS > 0:
        cmd += ["--max-hours", f"{ASR_MAX_HOURS:g}"]
    return cmd


def ensure_asr(video, group, args, root, suffix=""):
    """返回可用的 SRT 路径；识别失败返回 None。"""
    srt, jsonl = asr_files(root, group, video, suffix)
    if not args.force_asr and os.path.isfile(srt) and os.path.getsize(srt) > 0:
        log(f"    字幕复用 : {os.path.relpath(srt, root)}")
        return srt
    if args.no_asr:
        log("    ! 没有现成字幕，且指定了 --no-asr，跳过该视频")
        return None

    log(f"    识别中   : {os.path.basename(video)}")
    cmd = build_asr_cmd(video, srt, jsonl, args)
    t0 = time.time()
    code = subprocess.run(cmd, cwd=HERE).returncode
    if code != 0 or not os.path.isfile(srt):
        log(f"    ! 识别失败（退出码 {code}），跳过：{os.path.basename(video)}")
        return None
    log(f"    识别完成 : {os.path.basename(srt)}，耗时 {time.time() - t0:.0f}s")
    return srt


# ---------------------------------------------------------------------------
# 第 2 步：画面切换检测（ffmpeg scene，结果缓存）
# ---------------------------------------------------------------------------
def detect_cuts(video, limit_sec=0.0, threshold=SCENE_THRESHOLD, fps=SCENE_FPS):
    """返回画面切换时间点列表（秒）。含游戏内运镜/场景切换，不是严格剪辑点。"""
    tmp = tempfile.mkdtemp(prefix="cmp_scene_")
    name = "scene.txt"
    try:
        # metadata=print 写相对路径，避免 Windows 盘符里的冒号被 ffmpeg 当分隔符
        vf = (f"fps={fps},scale=320:180,"
              f"select=gt(scene\\,{threshold}),metadata=print:file={name}")
        cmd = [ewc.FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error", "-y"]
        if limit_sec and limit_sec > 0:
            cmd += ["-t", f"{float(limit_sec):.3f}"]
        cmd += ["-i", os.path.abspath(video), "-an", "-sn", "-dn",
                "-vf", vf, "-f", "null", os.devnull]
        subprocess.run(cmd, cwd=tmp, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        cuts = []
        path = os.path.join(tmp, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"pts_time:([0-9.]+)", line)
                    if m:
                        cuts.append(round(float(m.group(1)), 3))
        return sorted(set(cuts))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_cuts(video, group, args, root, limit_sec, suffix=""):
    if not args.cuts:
        return []
    stem = os.path.splitext(os.path.basename(video))[0] + suffix
    path = os.path.join(cache_dir(root, "cuts", group), stem + ".json")
    cfg = {"threshold": args.scene_threshold, "fps": SCENE_FPS,
           "limit_sec": round(float(limit_sec), 1)}
    cached = read_json(path)
    if isinstance(cached, dict) and cached.get("config") == cfg:
        return cached.get("cuts", [])
    cuts = detect_cuts(video, limit_sec, args.scene_threshold, SCENE_FPS)
    write_json(path, {"config": cfg, "cuts": cuts})
    return cuts


# ---------------------------------------------------------------------------
# 第 3 步：字幕 -> 段落 / 指标
# ---------------------------------------------------------------------------
def build_segments(entries, gap=SEG_GAP_SEC, min_sec=SEG_MIN_SEC, max_sec=SEG_MAX_SEC):
    """按停顿把字幕切成「段」：停顿 > gap 就断开。

    段 = 一段连续的说话内容（相当于一次没被打断的表达），
    用来比较两组成片的「每段时长」「每分钟段数」。
    """
    blocks = []
    for e in entries:
        if blocks and e["start"] - blocks[-1]["end"] <= gap:
            blocks[-1]["end"] = max(blocks[-1]["end"], e["end"])
            blocks[-1]["entries"].append(e)
        else:
            blocks.append({"start": e["start"], "end": e["end"], "entries": [e]})

    # 一直说个不停的块：在最大停顿处再切开（递归）
    split = []
    for b in blocks:
        split.extend(_split_long(b, max_sec))
    blocks = split

    # 太短的碎块并进前一段（间隔不大时），避免统计被一块两个字污染
    merged = []
    for b in blocks:
        if (merged and b["end"] - b["start"] < min_sec
                and b["start"] - merged[-1]["end"] <= gap * 2):
            merged[-1]["end"] = b["end"]
            merged[-1]["entries"].extend(b["entries"])
        else:
            merged.append(b)
    return merged


def _split_long(block, max_sec):
    if block["end"] - block["start"] <= max_sec:
        return [block]
    es = block["entries"]
    best_i, best_gap = None, -1.0
    for i in range(1, len(es)):
        g = es[i]["start"] - es[i - 1]["end"]
        if g > best_gap:
            best_gap, best_i = g, i
    if best_i is None or best_gap <= 0.01:
        return [block]
    left = {"start": block["start"], "end": es[best_i - 1]["end"],
            "entries": es[:best_i]}
    right = {"start": es[best_i]["start"], "end": block["end"],
             "entries": es[best_i:]}
    return _split_long(left, max_sec) + _split_long(right, max_sec)


def segment_record(block):
    texts = [e["text"] for e in block["entries"]]
    streamer = sum(e["end"] - e["start"] for e in block["entries"] if is_streamer(e))
    dur = max(block["end"] - block["start"], 1e-6)
    chars = sum(content_len(t) for t in texts)
    return {
        "start": round(block["start"], 3),
        "end": round(block["end"], 3),
        "dur": round(dur, 3),
        "n_subs": len(texts),
        "chars": chars,
        "streamer_sec": round(streamer, 3),
        "streamer_ratio": round(streamer / dur, 4),
        "text": " / ".join(texts),
    }


def style_features(entries):
    """文字风格指标（整片汇总后计算）。"""
    n = max(len(entries), 1)
    texts = [e["text"] for e in entries]
    joined = "".join(texts)
    cjk = _CJK_RE.findall(joined)
    n_chars = max(len(cjk), 1)

    q = sum(1 for t in texts if "？" in t)
    ex = sum(1 for t in texts if "！" in t)
    ell = sum(1 for t in texts if "…" in t)
    modal = sum(1 for ch in joined if ch in MODAL_CHARS)
    catch = sum(joined.count(w) for w in CATCHPHRASES)
    interact = sum(joined.lower().count(w) for w in INTERACT_WORDS)

    return {
        "问句率": 100.0 * q / n,            # 有问号的字幕占比（%）
        "感叹率": 100.0 * ex / n,
        "省略号率": 100.0 * ell / n,
        "语气词密度": 1000.0 * modal / n_chars,   # 每千字
        "口头禅密度": 1000.0 * catch / n_chars,
        "互动词密度": 1000.0 * interact / n_chars,
        "平均每条字数": sum(len(t) for t in texts) / n,
        "用字多样性": len(set(cjk)) / math.sqrt(n_chars),   # Guiraud 指数，与长度无关
        "第一人称密度": 1000.0 * joined.count("我") / n_chars,
        "第二人称密度": 1000.0 * joined.count("你") / n_chars,
    }


def analyze_transcript(video, group, entries, duration, duration_eff, cuts,
                       seg_cfg, srt_path):
    """把字幕变成一个视频的全套指标。"""
    segs = [segment_record(b) for b in build_segments(entries, *seg_cfg)]

    n = max(len(entries), 1)
    speech = sum(e["end"] - e["start"] for e in entries)
    streamer = sum(e["end"] - e["start"] for e in entries if is_streamer(e))
    chars = sum(content_len(e["text"]) for e in entries)

    gaps = [entries[i]["start"] - entries[i - 1]["end"] for i in range(1, len(entries))]
    gaps = [g for g in gaps if g > 0.2]
    long_pauses = [g for g in gaps if g > 2.0]
    seg_durs = [s["dur"] for s in segs]

    style = style_features(entries)
    m = {
        "group": group,
        "name": os.path.basename(video),
        "short": short_name(video),
        "path": video,
        "srt": srt_path,
        "duration": round(duration, 2),
        "duration_eff": round(duration_eff, 2),
        "n_subs": len(entries),
        "total_chars": chars,
        "speech_sec": round(speech, 2),
        "streamer_sec": round(streamer, 2),
        "other_sec": round(max(speech - streamer, 0.0), 2),
        "speech_ratio": round(pct(speech, duration_eff), 2),
        "streamer_ratio_of_speech": round(pct(streamer, speech), 2),
        "streamer_ratio_of_video": round(pct(streamer, duration_eff), 2),
        "subs_per_min": round(len(entries) / max(duration_eff / 60.0, 1e-6), 2),
        "chars_per_sec": round(chars / speech, 3) if speech > 0 else 0.0,
        "mean_sub_chars": round(sum(len(e["text"]) for e in entries) / n, 2),
        "median_pause": round(float(np.median(gaps)), 2) if gaps else 0.0,
        "long_pause_per_min": round(60.0 * len(long_pauses) / max(duration_eff / 60.0, 1e-6), 2),
        "seg_count": len(segs),
        "seg_per_10min": round(600.0 * len(segs) / max(duration_eff, 1e-6), 2),
        "seg_mean": round(float(np.mean(seg_durs)), 2) if seg_durs else 0.0,
        "seg_median": round(float(np.median(seg_durs)), 2) if seg_durs else 0.0,
        "seg_max": round(float(np.max(seg_durs)), 2) if seg_durs else 0.0,
        "seg_min": round(float(np.min(seg_durs)), 2) if seg_durs else 0.0,
        "seg_streamer_ratio": round(
            pct(sum(s["streamer_sec"] for s in segs), sum(s["dur"] for s in segs)), 2),
        "cut_count": len(cuts),
        "cut_per_10min": round(600.0 * len(cuts) / max(duration_eff, 1e-6), 2),
        "style": {k: round(v, 3) for k, v in style.items()},
        "segments": segs,
        "cuts": [round(c, 2) for c in cuts],
    }
    return m


def word_counts(entries):
    """jieba 分词计数（只留 2 字以上的实词，去掉纯数字 / 数字噪声）。"""
    try:
        import jieba
    except Exception:
        jieba = None

    cnt = Counter()
    for e in entries:
        text = e["text"]
        words = jieba.lcut(text) if jieba is not None else re.findall(r"[\u4e00-\u9fff]{2,4}", text)
        for w in words:
            w = w.strip()
            if len(w) < 2 or w in STOPWORDS:
                continue
            if not (_CJK_WORD_RE.match(w) or _ALNUM_RE.search(w)):
                continue
            if _CJK_WORD_RE.match(w) and len(w) > 6:
                continue
            digit_ratio = len(_DIGIT_RE.findall(w)) / len(w)
            if digit_ratio >= 0.5:      # 300、11、10 这类数字不算特征词
                continue
            cnt[w] += 1
    return cnt


def distinctive_words(cnt_a, cnt_b, n_a, n_b, top=14, min_total=8):
    """两组词频的 log-odds 对比：返回按 score 降序的 [(词, score, 次数a, 次数b)]。

    score > 0 表示 a 组（自动剪辑）更常用，< 0 表示 b 组（手动剪辑）更常用。
    """
    vocab = set(cnt_a) | set(cnt_b)
    la, lb = math.log(max(n_a, 1) + 1.0), math.log(max(n_b, 1) + 1.0)
    scored = []
    for w in vocab:
        ca, cb = cnt_a.get(w, 0), cnt_b.get(w, 0)
        if ca + cb < min_total:
            continue
        score = (math.log(ca + 0.5) - la) - (math.log(cb + 0.5) - lb)
        scored.append((w, score, ca, cb))
    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# 第 4 步：大模型（内容类型归类 + 总结）
# ---------------------------------------------------------------------------
TOPIC_SYSTEM = """你是游戏直播成片的内容分析员。输入是成片里按时间排列的段落，
每段有编号、时长、字数、主播说话占比和一段台词节选（用 / 分隔，可能被截断）。

请给每个段落判定一个「内容类型」，只能从下面列表里原样选一个：
{topic_list}

类型定义（严格按这个口径，尤其分清「杂谈闲聊」和「吐槽/玩梗」）：
{topic_guide}

只输出 JSON：{{"items": [{{"i": 编号, "topic": "类型"}}]}}，
每个编号都要有一条，不要输出其他字段。
"""

SUMMARY_SYSTEM = """你是资深游戏直播剪辑分析师。你会看到「手动剪辑」和「自动剪辑」
两组成片的统计指标（JSON），包括说话密度、主播说话比例、段落节奏、文字风格、
内容类型分布，以及每个视频的指标。

指标口径（务必按这个理解，不要自行发挥）：
- 语音密度% = 说话时长 / 视频时长
- 主播占语音% / 主播占整片% = 主播说话时长 / 总语音时长、/ 视频时长
- 段 = 字幕停顿超过 3 秒就断开的连续说话块；段长单位是秒
- 画面切换每 10 分钟：ffmpeg scene 打分（含游戏内运镜和转场），
  只作画面节奏参考，**不等于剪辑点**
- 文字风格：键名带 % 的是百分比，带 ‰ 的是每千字出现次数
- 内容类型占比%：大模型把段落归类后的占比（按段落时长加权）

请写一份中文对比分析，用 Markdown（不要整体套代码围栏），结构：
## 总体对比
（3-5 条，说明两组最明显的差异，每条都要带数字）
## 自动剪辑的特点
## 手动剪辑的特点
## 值得改进的地方
（给自动剪辑流程 2-3 条具体建议）

要求：只依据输入的数字下结论，不要编造；不要罗列所有指标；
总长度 450-750 字。只输出 JSON：{"markdown": "……"}。
"""

VIDEO_SYSTEM = """你是游戏直播剪辑分析师。你会看到一个成片的统计指标（JSON）。
指标口径：语音密度% = 说话时长/视频时长；主播占语音% = 主播说话占语音时长；
段 = 停顿超过 3 秒就断开的一段说话，段长单位是秒；画面切换每 10 分钟是 ffmpeg
scene 打分（含游戏内运镜），不等于剪辑点；文字风格里带 % 的是百分比、带 ‰ 的是每千字。

请用中文写 2-4 句、80-150 字的要点点评：这个视频的节奏、主播说话比例、
文字风格、内容偏好各有什么特点，和其他视频比有什么不一样。

只依据输入数据，不要编造。只输出 JSON：{"markdown": "……"}。
"""


def llm_json(system, user, model):
    obj = sv.parse_json_object(sv.llm_chat(system, user, model=model))
    if not isinstance(obj, dict):
        raise ValueError("大模型返回的不是 JSON 对象")
    return obj


def _excerpt(text, head=110, tail=40):
    text = (text or "").strip()
    if len(text) <= head + tail:
        return text
    return text[:head] + "……" + text[-tail:]


def _topic_rows(segs):
    rows = []
    for i, s in enumerate(segs):
        rows.append(
            f"#{i} {s['start']:.0f}-{s['end']:.0f}s（{s['dur']:.0f}s，"
            f"{s['chars']}字，主播说话 {s['streamer_ratio'] * 100:.0f}%）"
            f" 台词：{_excerpt(s['text'])}")
    return rows


def llm_topics(segs, cache_path, model):
    """给段落打内容类型标签；结果按起点缓存，重复跑不重复花钱。"""
    cached = read_json(cache_path, {}) or {}
    key_of = lambda s: f"{s['start']:.1f}"

    todo = [s for s in segs
            if s["dur"] >= LLM_TOPIC_MIN_SEC and s["chars"] >= 10
            and key_of(s) not in cached]
    if len(todo) > LLM_MAX_SEGMENTS_PER_VIDEO:
        step = len(todo) / float(LLM_MAX_SEGMENTS_PER_VIDEO)
        todo = [todo[int(i * step)] for i in range(LLM_MAX_SEGMENTS_PER_VIDEO)]

    system = TOPIC_SYSTEM.replace("{topic_list}", "、".join(TOPICS)) \
                          .replace("{topic_guide}", TOPIC_GUIDE)
    for i in range(0, len(todo), LLM_TOPIC_BATCH):
        batch = todo[i:i + LLM_TOPIC_BATCH]
        user = "段落列表：\n" + "\n".join(_topic_rows(batch))
        try:
            obj = llm_json(system, user, model)
        except Exception as exc:
            log(f"    ! 内容归类失败（{len(batch)} 段）：{exc}")
            continue
        for item in obj.get("items", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            topic = str(item.get("topic", "")).strip()
            if not (0 <= idx < len(batch)) or topic not in TOPICS:
                continue
            cached[key_of(batch[idx])] = topic

    write_json(cache_path, cached)
    for s in segs:
        s["topic"] = cached.get(key_of(s), "")
    return cached


def topic_shares(segs):
    """{内容类型: 秒数}，只统计已归类的段。"""
    out = Counter()
    for s in segs:
        if s.get("topic"):
            out[s["topic"]] += s["dur"]
    return out


def _compact_video(m):
    return {
        "视频": m["name"],
        "时长分钟": round(m["duration_eff"] / 60, 1),
        "语音密度%": m["speech_ratio"],
        "主播占语音%": m["streamer_ratio_of_speech"],
        "段数": m["seg_count"],
        "段长中位秒": m["seg_median"],
        "语速字每秒": m["chars_per_sec"],
        "问句率%": m["style"]["问句率"],
        "口头禅密度‰": m["style"]["口头禅密度"],
        "内容类型": _top_topics(m, 3),
    }


def _top_topics(m, k=3):
    shares = topic_shares(m.get("segments", []))
    total = sum(shares.values()) or 1
    return {t: round(100.0 * s / total, 1) for t, s in shares.most_common(k)}


def _llm_group_payload(a):
    """给大模型看的分组指标：键名自带单位，避免它把秒/百分比说错。"""
    return {
        "视频数": a["n_videos"],
        "总时长分钟": round(a["duration"] / 60.0, 1),
        "平均单视频分钟": round(a["mean_duration"] / 60.0, 1),
        "语音密度%": round(a["speech_ratio"], 1),
        "主播占语音%": round(a["streamer_ratio_of_speech"], 1),
        "主播占整片%": round(a["streamer_ratio_of_video"], 1),
        "段数": a["seg_count"],
        "每10分钟段数": round(a["seg_per_10min"], 1),
        "段长中位秒": round(a["seg_median"], 0),
        "段长平均秒": round(a["seg_mean"], 0),
        "最长段秒": round(a["seg_max"], 0),
        "段内主播占比%": round(a["seg_streamer_ratio"], 1),
        "画面切换每10分钟": round(a["cut_per_10min"], 1),
        "字幕条数": a["n_subs"],
        "语速字每秒": round(a["chars_per_sec"], 2),
        "文字风格": a["style"],
        "内容类型占比%": a["topics"],
    }


def llm_summary(group_stats, video_metrics, model):
    system = SUMMARY_SYSTEM
    user = ("统计指标（JSON）：\n" + json.dumps(
        {"分组统计": {g: _llm_group_payload(a) for g, a in group_stats.items()},
         "每个视频": [_compact_video(m) for m in video_metrics]},
        ensure_ascii=False, indent=1))
    try:
        obj = llm_json(system, user, model)
        return str(obj.get("markdown", "")).strip()
    except Exception as exc:
        log(f"  ! 总体总结生成失败：{exc}")
        return ""


def llm_video_note(m, model):
    user = "指标（JSON）：\n" + json.dumps(_compact_video(m), ensure_ascii=False)
    try:
        obj = llm_json(VIDEO_SYSTEM, user, model)
        return str(obj.get("markdown", "")).strip()
    except Exception as exc:
        log(f"    ! 视频点评生成失败（{m['name']}）：{exc}")
        return ""


# ---------------------------------------------------------------------------
# 第 5 步：图表
# ---------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", "PingFang SC", "Arial Unicode MS"):
        if cand in names:
            matplotlib.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 110
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax):
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _boxplot(ax, data, labels=None, horizontal=True, colors=None, widths=0.6):
    """boxplot 兼容层：新版 matplotlib 用 orientation，老版用 vert。"""
    kw = {"showfliers": False, "patch_artist": True, "widths": widths}
    try:
        bp = ax.boxplot(data, orientation="horizontal" if horizontal else "vertical", **kw)
    except TypeError:
        bp = ax.boxplot(data, vert=not horizontal, **kw)
    if colors is not None:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("#222222")
    if labels is not None:
        if horizontal:
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels)
        else:
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
    return bp


def chart_overview(videos, path, plt):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("图 1  成片结构概览（上：语音密度 / 说话构成；下：段长 / 段落密度）",
                 fontsize=13)

    # (a) 语音密度
    ax = axes[0][0]
    vs = list(reversed(videos))
    vals = [v["speech_ratio"] for v in vs]
    ax.barh([v["short"] for v in vs], vals,
            color=[GROUP_COLOR[v["group"]] for v in vs], alpha=0.85)
    for i, val in enumerate(vals):
        ax.text(val + 1.5, i, f"{val:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel("语音密度 = 说话时长 / 视频时长（%）")
    ax.set_title("语音密度：剪掉静场越多，数值越高")
    _style_axes(ax)

    # (b) 主播 / 其他 / 静音 构成
    ax = axes[0][1]
    labels = [v["short"] for v in vs]
    for i, v in enumerate(vs):
        dur = max(v["duration_eff"], 1e-6)
        seg1 = 100.0 * v["streamer_sec"] / dur
        seg2 = 100.0 * v["other_sec"] / dur
        seg3 = max(100.0 - seg1 - seg2, 0.0)
        left = 0.0
        for val, color, label in ((seg1, STREAMER_COLOR, "主播说话"),
                                  (seg2, OTHER_COLOR, "其他人说话（游戏/嘉宾）"),
                                  (seg3, "#D9D9D9", "静音 / 无字幕")):
            ax.barh(i, val, left=left, color=color,
                    label=label if i == 0 else None, height=0.7)
            left += val
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占视频时长比例（%）")
    ax.set_title("说话构成：主播 vs 其他声音 vs 静场")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8,
              ncol=3, frameon=False)

    # (c) 每段时长箱线图
    ax = axes[1][0]
    ok = [v for v in vs if v["segments"]]
    data = [[s["dur"] for s in v["segments"]] for v in ok]
    if data:
        _boxplot(ax, data, labels=[v["short"] for v in ok],
                 colors=[GROUP_COLOR[v["group"]] for v in ok])
    ax.set_xlabel("每段时长（秒，按停顿 > %.1fs 划分）" % SEG_GAP_SEC)
    ax.set_title("每段时长分布")
    _style_axes(ax)

    # (d) 段落密度
    ax = axes[1][1]
    vals = [v["seg_per_10min"] for v in vs]
    ax.barh([v["short"] for v in vs], vals,
            color=[GROUP_COLOR[v["group"]] for v in vs], alpha=0.85)
    for i, val in enumerate(vals):
        ax.text(val + 0.2, i, f"{val:.1f}", va="center", fontsize=8)
    ax.set_xlabel("每 10 分钟段数（数值大 = 切得碎、节奏快）")
    ax.set_title("段落密度")
    _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_segments(videos, path, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("图 2  段落时长分布对比", fontsize=13)

    per_group = {g: [s["dur"] for v in videos if v["group"] == g for s in v["segments"]]
                 for g in ("manual", "auto")}
    all_durs = [d for ds in per_group.values() for d in ds]
    if not all_durs:
        return
    hi = max(20.0, float(np.percentile(all_durs, 97)))

    ax = axes[0]
    bins = np.linspace(0, hi, 30)
    for g, ds in per_group.items():
        if ds:
            ax.hist(ds, bins=bins, alpha=0.55, color=GROUP_COLOR[g],
                    label=f"{GROUP_LABEL[g]}（{len(ds)} 段）", density=True)
    ax.set_xlabel("段长（秒）")
    ax.set_ylabel("概率密度")
    ax.set_title("段长分布（归一化）")
    ax.legend(fontsize=9)
    _style_axes(ax)

    ax = axes[1]
    for g, ds in per_group.items():
        if not ds:
            continue
        x = np.sort(np.asarray(ds, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x) * 100.0
        ax.plot(x, y, color=GROUP_COLOR[g], label=GROUP_LABEL[g], linewidth=1.8)
    ax.set_xlim(0, hi)
    ax.set_xlabel("段长（秒）")
    ax.set_ylabel("累计占比（%）")
    ax.set_title("累计分布：曲线越靠左，段越短")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    data, labels, colors = [], [], []
    for g in ("manual", "auto"):
        if per_group[g]:
            data.append(per_group[g])
            labels.append(GROUP_LABEL[g])
            colors.append(GROUP_COLOR[g])
    if data:
        _boxplot(ax, data, labels=labels, horizontal=False, colors=colors, widths=0.5)
    ax.set_ylabel("段长（秒）")
    ax.set_title("两组段长总体对比")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


STYLE_AXES = [("问句率", "问句率", "style"), ("感叹率", "感叹率", "style"),
              ("语气词密度", "语气词密度", "style"), ("口头禅密度", "口头禅密度", "style"),
              ("互动词密度", "互动词密度", "style"),
              ("streamer_ratio_of_speech", "主播占语音", "top")]


def chart_style(agg, path, plt):
    keys = [label for _, label, _ in STYLE_AXES]
    values = {}
    for g in ("manual", "auto"):
        a = agg.get(g)
        if not a:
            continue
        values[g] = [a["style"].get(k, 0.0) if src == "style" else a.get(k, 0.0)
                     for k, _, src in STYLE_AXES]
    if not values:
        return

    maxima = [max(values[g][i] for g in values) or 1.0 for i in range(len(keys))]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, polar=True)
    for g, vals in values.items():
        norm = [v / m for v, m in zip(vals, maxima)]
        norm += norm[:1]
        ax.plot(angles, norm, color=GROUP_COLOR[g], linewidth=2, label=GROUP_LABEL[g])
        ax.fill(angles, norm, color=GROUP_COLOR[g], alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{k}\n最大值 {m:.1f}" for k, m in zip(keys, maxima)], fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title("图 3  文字风格雷达图（各轴按两组最大值归一化）", pad=22, fontsize=13)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def placeholder_chart(path, plt, text):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_words(scored, path, plt, top=14):
    auto_words = [s for s in scored if s[1] > 0][:top]
    manual_words = [s for s in scored if s[1] < 0][-top:][::-1]
    if not auto_words and not manual_words:
        placeholder_chart(path, plt,
                          "字幕样本太少，还没形成两组之间的特征词差异。\n"
                          "（整片跑完、每组字幕上千条后会有内容）")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("图 4  两组的特征词（jieba 词频 + log-odds 对比）", fontsize=13)

    for ax, words, group, title in (
            (axes[0], manual_words, "manual", "手动剪辑更常说"),
            (axes[1], auto_words, "auto", "自动剪辑更常说")):
        if not words:
            ax.axis("off")
            continue
        labels = [w for w, _, _, _ in words][::-1]
        counts = [(ca if group == "auto" else cb) for _, _, ca, cb in words][::-1]
        ax.barh(labels, counts, color=GROUP_COLOR[group], alpha=0.85)
        for i, c in enumerate(counts):
            ax.text(c, i, f" {c}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("出现次数")
        ax.margins(x=0.15)
        _style_axes(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_topics(videos, path, plt):
    have = [v for v in videos if any(s.get("topic") for s in v["segments"])]
    if not have:
        return

    total = Counter()
    for v in have:
        total.update(topic_shares(v["segments"]))
    order = [t for t, _ in total.most_common()]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(order), 2)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.6 * len(have) + 3),
                             gridspec_kw={"width_ratios": [1, 1.7]})
    fig.suptitle("图 5  内容类型分布（大模型归类，按段落时长加权）", fontsize=13)

    # 左：分组汇总
    ax = axes[0]
    for i, g in enumerate(("manual", "auto")):
        shares = Counter()
        for v in have:
            if v["group"] == g:
                shares.update(topic_shares(v["segments"]))
        s = sum(shares.values()) or 1
        left = 0.0
        for j, t in enumerate(order):
            val = 100.0 * shares.get(t, 0) / s
            ax.barh(i, val, left=left, color=colors[j], height=0.5)
            if val >= 6:
                ax.text(left + val / 2, i, f"{val:.0f}%", ha="center",
                        va="center", fontsize=8, color="#111111")
            left += val
    ax.set_yticks([0, 1])
    ax.set_yticklabels([GROUP_LABEL["manual"], GROUP_LABEL["auto"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("占比（%）")
    ax.set_title("分组汇总")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(len(order))],
              labels=order, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))

    # 右：每个视频
    ax = axes[1]
    rows = list(reversed(have))
    for i, v in enumerate(rows):
        shares = topic_shares(v["segments"])
        s = sum(shares.values()) or 1
        left = 0.0
        for j, t in enumerate(order):
            val = 100.0 * shares.get(t, 0) / s
            ax.barh(i, val, left=left, color=colors[j], height=0.7)
            left += val
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([v["short"] for v in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("占比（%）")
    ax.set_title("每个视频")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_timeline(videos, path, plt):
    if not videos:
        return
    max_min = max(v["duration_eff"] / 60.0 for v in videos) or 1.0
    fig, ax = plt.subplots(figsize=(15, 0.55 * len(videos) + 2.2))
    fig.suptitle("图 6  说话时间线（红=主播说话，蓝=其他声音，空白=静场）", fontsize=13)

    rows = list(reversed(videos))
    for i, v in enumerate(rows):
        for e in v["entries"]:
            x0 = e["start"] / 60.0
            w = max((e["end"] - e["start"]) / 60.0, 0.02)
            ax.broken_barh([(x0, w)], (i - 0.32, 0.64),
                           color=STREAMER_COLOR if is_streamer(e) else OTHER_COLOR,
                           linewidth=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{v['short']}" for v in rows])
    ax.set_xlim(0, max_min * 1.01)
    ax.set_xlabel("视频内时间（分钟）")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=STREAMER_COLOR),
                       plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR)],
              labels=["主播说话", "其他声音"], loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 汇总 + 文字报告
# ---------------------------------------------------------------------------
def aggregate(group, vids):
    """把一个组的所有视频汇总成一组指标（比率用总量算，风格用全部字幕算）。"""
    entries = [e for v in vids for e in v["entries"]]
    segs = [s for v in vids for s in v["segments"]]
    dur = sum(v["duration_eff"] for v in vids)
    speech = sum(v["speech_sec"] for v in vids)
    streamer = sum(v["streamer_sec"] for v in vids)
    chars = sum(v["total_chars"] for v in vids)
    seg_durs = [s["dur"] for s in segs]
    cuts = sum(v["cut_count"] for v in vids)

    style = style_features(entries)
    shares = Counter()
    for v in vids:
        shares.update(topic_shares(v["segments"]))

    return {
        "group": group,
        "n_videos": len(vids),
        "duration": dur,
        "mean_duration": dur / max(len(vids), 1),
        "speech_ratio": pct(speech, dur),
        "streamer_ratio_of_speech": pct(streamer, speech),
        "streamer_ratio_of_video": pct(streamer, dur),
        "n_subs": len(entries),
        "subs_per_min": len(entries) / max(dur / 60.0, 1e-6),
        "chars_per_sec": chars / speech if speech else 0.0,
        "mean_sub_chars": sum(len(e["text"]) for e in entries) / max(len(entries), 1),
        "seg_count": len(segs),
        "seg_per_10min": 600.0 * len(segs) / max(dur, 1e-6),
        "seg_median": float(np.median(seg_durs)) if seg_durs else 0.0,
        "seg_mean": float(np.mean(seg_durs)) if seg_durs else 0.0,
        "seg_max": float(np.max(seg_durs)) if seg_durs else 0.0,
        "seg_streamer_ratio": pct(sum(s["streamer_sec"] for s in segs),
                                  sum(s["dur"] for s in segs)),
        "cut_per_10min": 600.0 * cuts / max(dur, 1e-6),
        "style": {k: round(v, 2) for k, v in style.items()},
        "topics": {t: round(pct(s, sum(shares.values())), 1)
                   for t, s in shares.most_common()},
        "topic_sec": dict(shares),
    }


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt_num(x, nd=1):
    return f"{float(x):.{nd}f}"


def rule_summary(agg, videos):
    """不依赖大模型的规则版总结（有 key 时也会保留，作为数据支撑部分）。"""
    a, m = agg.get("auto"), agg.get("manual")
    if not a or not m:
        return ["- 两组视频不全，无法对比。"]

    out = []
    d = a['speech_ratio'] - m['speech_ratio']
    if d > 3:
        tail = "自动成片明显更紧凑，静场被大量剪掉。"
    elif d < -3:
        tail = "自动成片的语音密度反而更低，保留的静场 / 长镜头比手动多。"
    else:
        tail = "两组紧凑程度接近。"
    out.append(f"- **语音密度**：自动剪辑 {fmt_num(a['speech_ratio'])}%，"
               f"手动剪辑 {fmt_num(m['speech_ratio'])}%"
               f"（相差 {d:+.1f} 个百分点）。{tail}")

    da = a['streamer_ratio_of_speech'] - m['streamer_ratio_of_speech']
    if da > 2:
        tail = "自动成片更偏向只留主播说话，非主播的声音（游戏原声、过场演出）剪得更狠。"
    elif da < -2:
        tail = ("自动成片混进了更多非主播的语音（游戏原声 / 剧情演出 / 其他人），"
                "手动剪辑基本只留主播。")
    else:
        tail = "两组主播占比接近。"
    out.append(f"- **主播说话占比**：按语音时长算，自动 {fmt_num(a['streamer_ratio_of_speech'])}%、"
               f"手动 {fmt_num(m['streamer_ratio_of_speech'])}%；"
               f"按整条视频算，自动 {fmt_num(a['streamer_ratio_of_video'])}%、"
               f"手动 {fmt_num(m['streamer_ratio_of_video'])}%。{tail}")

    out.append(f"- **段落节奏**：每 10 分钟自动 {fmt_num(a['seg_per_10min'])} 段、"
               f"手动 {fmt_num(m['seg_per_10min'])} 段；段长中位数自动 "
               f"{fmt_num(a['seg_median'], 0)}s、手动 {fmt_num(m['seg_median'], 0)}s；"
               f"最长段自动 {fmt_num(a['seg_max'], 0)}s、手动 {fmt_num(m['seg_max'], 0)}s。"
               + ("自动剪辑切得更碎、单段更短。" if a['seg_median'] < m['seg_median'] else
                  "自动剪辑的单段反而更长，切点更少。"))

    sa, sm = a["style"], m["style"]
    out.append(f"- **文字风格**：问句率自动 {fmt_num(sa['问句率'])}% / 手动 {fmt_num(sm['问句率'])}%；"
               f"感叹率 {fmt_num(sa['感叹率'])}% / {fmt_num(sm['感叹率'])}%；"
               f"语气词密度 {fmt_num(sa['语气词密度'])}‰ / {fmt_num(sm['语气词密度'])}‰；"
               f"口头禅密度 {fmt_num(sa['口头禅密度'])}‰ / {fmt_num(sm['口头禅密度'])}‰；"
               f"语速 {a['chars_per_sec']:.2f} / {m['chars_per_sec']:.2f} 字每秒；"
               f"用字多样性 {fmt_num(sa['用字多样性'], 2)} / {fmt_num(sm['用字多样性'], 2)}。")
    ia, im = sa["互动词密度"], sm["互动词密度"]
    if max(ia, im) < 0.5:
        tail = ("两组字幕里几乎没出现「弹幕 / 观众 / 兄弟」这类词，"
                "这项指标参考价值有限。")
    elif ia < im * 0.9:
        tail = "自动剪辑确实更少出现读弹幕的内容（和提示词里「避免读弹幕」一致）。"
    elif ia > im * 1.1:
        tail = "自动剪辑里的读弹幕 / 和观众对话反而更多。"
    else:
        tail = "两组的互动内容密度差别不大。"
    out.append(f"- **和观众的互动**：互动词密度自动 {fmt_num(ia)}‰ / 手动 {fmt_num(im)}‰。{tail}")

    if a["topics"] and m["topics"]:
        ta = "、".join(f"{t} {p}%" for t, p in list(a["topics"].items())[:3])
        tm = "、".join(f"{t} {p}%" for t, p in list(m["topics"].items())[:3])
        out.append(f"- **内容偏好（大模型归类）**：自动剪辑以 {ta} 为主；手动剪辑以 {tm} 为主。")

    out.append(f"- **画面切换**：每 10 分钟自动 {fmt_num(a['cut_per_10min'])} 次、"
               f"手动 {fmt_num(m['cut_per_10min'])} 次"
               "（含游戏内运镜/场景切换，不等于剪辑点，只作节奏参考）。")
    return out


def video_note_rule(m):
    """规则版单视频点评（大模型不可用时用）。"""
    st = m["style"]
    note = (f"{fmt_num(m['duration_eff'] / 60)} 分钟，语音密度 {fmt_num(m['speech_ratio'])}%，"
            f"主播说话占语音 {fmt_num(m['streamer_ratio_of_speech'])}%；"
            f"共 {m['seg_count']} 段，段长中位 {fmt_num(m['seg_median'], 0)}s、"
            f"最长 {fmt_num(m['seg_max'], 0)}s；"
            f"语速 {m['chars_per_sec']:.2f} 字/秒，问句率 {fmt_num(st['问句率'])}%、"
            f"口头禅密度 {fmt_num(st['口头禅密度'])}‰。")
    top = _top_topics(m, 3)
    if top:
        note += " 内容以 " + "、".join(f"{t} {p}%" for t, p in top.items()) + " 为主。"
    return note


def build_markdown(args, videos, agg, scored, llm_text, video_notes, charts):
    lines = []
    total_dur = sum(v["duration_eff"] for v in videos)
    section = iter("一二三四五六七八九十")

    def sec(title):
        lines.append(f"## {next(section)}、{title}")
        lines.append("")

    lines.append("# 手动剪辑 vs 自动剪辑：对比报告")
    lines.append("")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 样本：手动剪辑 {agg.get('manual', {}).get('n_videos', 0)} 个视频 / "
                 f"自动剪辑 {agg.get('auto', {}).get('n_videos', 0)} 个视频，"
                 f"合计 {fmt_min(total_dur)} 分钟")
    lines.append(f"- 字幕：`sensevoice_srt.py`（SenseVoice + 大模型纠错"
                 f"{' + 主播声纹标注' if args.mark_streamer else ''}）")
    lines.append(f"- 段落划分：停顿 > {SEG_GAP_SEC:g}s 记为分段；"
                 f"画面切换阈值 {args.scene_threshold:g}"
                 f"{'（已关闭）' if not args.cuts else ''}")
    if args.limit_minutes > 0:
        lines.append(f"- ⚠️ 试跑模式：每个视频只分析了前 {args.limit_minutes:g} 分钟")
    lines.append("")

    if llm_text:
        sec("分析结论（大模型）")
        lines.append(llm_text)
        lines.append("")
    else:
        sec("分析结论")
        lines.append("（未调用大模型生成结论文本；设置 `DEEPSEEK_API_KEY` "
                     "并去掉 `--no-llm` 后重跑即可生成。下面的数据要点是规则版总结。）")
        lines.append("")
    sec("数据要点")
    lines.extend(rule_summary(agg, videos))
    lines.append("")

    sec("核心指标对比")
    a, m = agg.get("auto", {}), agg.get("manual", {})
    rows = [
        ("视频数（个）", f"{m.get('n_videos', 0)}", f"{a.get('n_videos', 0)}"),
        ("总时长（分钟）", fmt_min(m.get("duration", 0)), fmt_min(a.get("duration", 0))),
        ("平均单条时长（分钟）", fmt_min(m.get("mean_duration", 0)),
         fmt_min(a.get("mean_duration", 0))),
        ("语音密度（%）", fmt_num(m.get("speech_ratio", 0)), fmt_num(a.get("speech_ratio", 0))),
        ("主播说话 / 语音（%）", fmt_num(m.get("streamer_ratio_of_speech", 0)),
         fmt_num(a.get("streamer_ratio_of_speech", 0))),
        ("主播说话 / 视频（%）", fmt_num(m.get("streamer_ratio_of_video", 0)),
         fmt_num(a.get("streamer_ratio_of_video", 0))),
        ("字幕条数", f"{m.get('n_subs', 0)}", f"{a.get('n_subs', 0)}"),
        ("每分钟字幕条数", fmt_num(m.get("subs_per_min", 0)), fmt_num(a.get("subs_per_min", 0))),
        ("平均每条字幕字数", fmt_num(m.get("mean_sub_chars", 0)), fmt_num(a.get("mean_sub_chars", 0))),
        ("语速（字/秒）", f"{m.get('chars_per_sec', 0):.2f}", f"{a.get('chars_per_sec', 0):.2f}"),
        ("段数（合计）", f"{m.get('seg_count', 0)}", f"{a.get('seg_count', 0)}"),
        ("每 10 分钟段数", fmt_num(m.get("seg_per_10min", 0)), fmt_num(a.get("seg_per_10min", 0))),
        ("段长中位数（秒）", fmt_num(m.get("seg_median", 0), 0), fmt_num(a.get("seg_median", 0), 0)),
        ("段长平均（秒）", fmt_num(m.get("seg_mean", 0), 0), fmt_num(a.get("seg_mean", 0), 0)),
        ("最长段（秒）", fmt_num(m.get("seg_max", 0), 0), fmt_num(a.get("seg_max", 0), 0)),
        ("段内主播说话占比（%）", fmt_num(m.get("seg_streamer_ratio", 0)),
         fmt_num(a.get("seg_streamer_ratio", 0))),
        ("画面切换 / 10 分钟", fmt_num(m.get("cut_per_10min", 0)), fmt_num(a.get("cut_per_10min", 0))),
    ]
    for key, label, unit in (
            ("问句率", "问句率（%）", ""), ("感叹率", "感叹率（%）", ""),
            ("省略号率", "省略号率（%）", ""), ("语气词密度", "语气词密度（‰）", ""),
            ("口头禅密度", "口头禅密度（‰）", ""), ("互动词密度", "互动词密度（‰）", ""),
            ("平均每条字数", "字幕平均字数", ""), ("用字多样性", "用字多样性（Guiraud）", ""),
            ("第一人称密度", "第一人称密度（‰）", ""),
            ("第二人称密度", "第二人称密度（‰）", "")):
        rows.append((label,
                     fmt_num(m.get("style", {}).get(key, 0), 2),
                     fmt_num(a.get("style", {}).get(key, 0), 2)))
    lines.append(md_table(["指标", "手动剪辑", "自动剪辑"], rows))
    lines.append("")

    sec("每个视频的指标")
    rows = []
    for v in videos:
        rows.append([
            GROUP_LABEL.get(v["group"], v["group"]),
            v["name"],
            fmt_min(v["duration_eff"]),
            fmt_num(v["speech_ratio"], 0),
            fmt_num(v["streamer_ratio_of_speech"], 0),
            v["seg_count"],
            fmt_num(v["seg_median"], 0),
            f"{v['chars_per_sec']:.2f}",
            fmt_num(v["style"]["问句率"], 1),
            fmt_num(v["style"]["口头禅密度"], 1),
        ])
    lines.append(md_table(["组", "视频", "时长(分)", "语音密度%", "主播占语音%",
                           "段数", "段长中位(s)", "语速(字/s)", "问句率%", "口头禅‰"], rows))
    lines.append("")

    sec("内容偏好")
    topics = []
    for g in ("manual", "auto"):
        for t, p in list(agg.get(g, {}).get("topics", {}).items()):
            topics.append([GROUP_LABEL[g], t, f"{p}%"])
    if topics:
        lines.append(md_table(["组", "内容类型", "占说话时长"], topics))
    else:
        lines.append("（未做内容归类：--no-llm 或没有 DEEPSEEK_API_KEY）")
    lines.append("")

    if scored:
        auto_w = [w for w, s, _, _ in scored if s > 0][:12]
        manual_w = [w for w, s, _, _ in scored if s < 0][-12:][::-1]
        sec("特征词")
        lines.append(f"- 自动剪辑更常说：{'、'.join(auto_w) if auto_w else '（无明显差异）'}")
        lines.append(f"- 手动剪辑更常说：{'、'.join(manual_w) if manual_w else '（无明显差异）'}")
        lines.append("")

    sec("逐片特点")
    for v in videos:
        lines.append(f"### {v['name']}（{GROUP_LABEL.get(v['group'], v['group'])}）")
        lines.append("")
        lines.append(video_note_rule(v))
        note = video_notes.get(v["name"], "")
        if note:
            lines.append("")
            lines.append(note)
        lines.append("")

    sec("图表")
    for name, title in charts:
        lines.append(f"![{title}]({name})")
        lines.append("")

    sec("数据和口径说明")
    lines.append(f"- 「段」= 字幕停顿 > {SEG_GAP_SEC:g} 秒就断开的连续说话块"
                 f"（<{SEG_MIN_SEC:g}s 的碎块并入相邻段，单段超过 {SEG_MAX_SEC:g}s 会在最大停顿处再切）；"
                 "这是从字幕反推的表达结构，不等于精确到帧的剪辑点。")
    lines.append("- 「主播」来自 sensevoice_srt.py 的声纹二分类标注（（主播）前缀），"
                 "阈值和参考样本会影响绝对数值，建议两组用同一套参数（本报告已统一）。")
    lines.append("- 「用字多样性」用 Guiraud 指数（不同字数 / √总字数），不受成片长短影响；"
                 "数值越大说明用词越分散、重复越少。")
    lines.append("- 「画面切换」是 ffmpeg scene 打分（含游戏内运镜、转场、切场景），"
                 "不能等同于人工/AI 的剪辑点。")
    lines.append("- 两组的素材本身不是同一批直播（游戏和日期都不同），"
                 "所以差异里既有「剪辑习惯」也有「内容本身」的成分，读结论时请留意。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="手动剪辑 vs 自动剪辑：字幕 + 结构 + 文字风格对比，出图表和报告")
    p.add_argument("--manual", default=MANUAL_DIR, help="手动剪辑视频文件夹")
    p.add_argument("--auto", default=AUTO_DIR, help="自动剪辑视频文件夹")
    p.add_argument("-o", "--out", default=REPORT_DIR, help="报告输出目录")
    p.add_argument("--only", default="", help="只处理文件名包含该字符串的视频（调试）")

    g = p.add_argument_group("字幕（sensevoice_srt.py）")
    g.add_argument("--device", default=ASR_DEVICE, help="auto / cuda:0 / cpu")
    g.add_argument("--language", default=ASR_LANGUAGE, help="zh / en / yue / ja / ko / auto")
    g.add_argument("--no-asr", dest="no_asr", action="store_true",
                   help="只用已有字幕，不调用 sensevoice_srt.py")
    g.add_argument("--force-asr", action="store_true", help="已有字幕也重新识别")
    g.add_argument("--no-correct", dest="correct", action="store_false",
                   default=ASR_CORRECT, help="关闭字幕纠错（少一轮大模型）")
    g.add_argument("--no-streamer-mark", dest="mark_streamer",
                   action="store_false", default=ASR_MARK_STREAMER,
                   help="关闭主播声纹标注（主播相关指标会全为 0）")
    g.add_argument("--streamer-ref", default=ASR_STREAMER_REF,
                   help="主播声纹参考样本（音频或视频），空 = 自动注册")
    g.add_argument("--streamer-threshold", type=float, default=ASR_STREAMER_THRESHOLD,
                   help="主播声纹阈值，0 = 用 sensevoice_srt.py 的默认值")
    g.add_argument("--ref-sec", type=float, default=ASR_REF_SEC,
                   help="参考样本最多用前多少秒，0 = 默认")
    g.add_argument("--limit-minutes", type=float, default=0.0,
                   help="试跑：每个视频只识别前 N 分钟（缓存加后缀，不会影响正式结果）")

    g = p.add_argument_group("分析")
    g.add_argument("--seg-gap", type=float, default=SEG_GAP_SEC, help="分段停顿阈值（秒）")
    g.add_argument("--seg-min", type=float, default=SEG_MIN_SEC, help="最短段（秒）")
    g.add_argument("--seg-max", type=float, default=SEG_MAX_SEC, help="最长段（秒）")
    g.add_argument("--no-cuts", dest="cuts", action="store_false", default=SCENE_DETECT,
                   help="不做画面切换检测（省时间）")
    g.add_argument("--scene-threshold", type=float, default=SCENE_THRESHOLD,
                   help="ffmpeg scene 阈值")

    g = p.add_argument_group("大模型")
    g.add_argument("--no-llm", dest="llm", action="store_false", default=LLM_SUMMARY,
                   help="不调大模型（只出规则版总结）")
    g.add_argument("--llm-model", default=sv.CORRECT_MODEL, help="大模型名字")

    return p.parse_args()


def collect_videos(folder, pattern):
    paths = ewc.list_files(folder, ewc.VIDEO_EXTS)
    if pattern:
        key = pattern.lower()
        paths = [p for p in paths if key in os.path.basename(p).lower()]
    return paths


def main():
    # 控制台是 GBK（PowerShell 默认）时保持原样，中文才不会乱码；
    # 只有在 ASCII / 其他窄编码的终端里才切到 UTF-8。
    try:
        enc = (sys.stdout.encoding or "").lower().replace("-", "")
        if enc in ("ascii", "latin1", "cp1252"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    ewc.require_tools()

    root = ensure_dir(args.out)
    suffix = variant_suffix(args.limit_minutes)
    seg_cfg = (args.seg_gap, args.seg_min, args.seg_max)

    llm_ok = bool(args.llm) and bool(sv.llm_api_key())
    if args.llm and not llm_ok:
        log(f"  ! 未设置 {sv.CORRECT_API_KEY_ENV}，跳过大模型归类 / 总结")
        log(f"    （PowerShell: $env:{sv.CORRECT_API_KEY_ENV}=\"sk-xxxx\"）")

    log("=" * 72)
    log("手动剪辑 vs 自动剪辑 对比")
    log(f"  手动: {args.manual}")
    log(f"  自动: {args.auto}")
    log(f"  输出: {root}")
    if args.limit_minutes > 0:
        log(f"  ! 试跑模式：每个视频只分析前 {args.limit_minutes:g} 分钟")
    log("=" * 72)

    videos, skipped = [], []
    t0 = time.time()

    for group, folder in (("manual", args.manual), ("auto", args.auto)):
        paths = collect_videos(folder, args.only)
        log(f"\n=== {GROUP_LABEL[group]}：{len(paths)} 个视频 ===")
        if not paths:
            log(f"  ! {folder} 里没有找到视频")
            continue

        for i, path in enumerate(paths, 1):
            name = os.path.basename(path)
            log(f"\n[{i}/{len(paths)}] {name}")
            srt = ensure_asr(path, group, args, root, suffix)
            if not srt:
                skipped.append(path)
                continue

            try:
                entries = ewc.read_srt(srt)
            except Exception as exc:
                log(f"  ! 字幕解析失败：{exc}")
                skipped.append(path)
                continue
            if not entries:
                log("  ! 字幕是空的，跳过")
                skipped.append(path)
                continue

            duration = ewc.probe_duration(path)
            limit = args.limit_minutes * 60.0 if args.limit_minutes > 0 else 0.0
            duration_eff = min(duration, limit) if limit > 0 else duration
            entries = [e for e in entries if e["start"] < duration_eff]

            cuts = ensure_cuts(path, group, args, root, duration_eff, suffix)

            m = analyze_transcript(path, group, entries, duration, duration_eff,
                                   cuts, seg_cfg, srt)
            m["entries"] = entries

            log(f"    时长 {fmt_min(duration_eff)} 分钟｜语音密度 "
                f"{m['speech_ratio']:.0f}%｜主播占语音 "
                f"{m['streamer_ratio_of_speech']:.0f}%｜{m['seg_count']} 段"
                f"（中位 {m['seg_median']:.0f}s）｜切换 {m['cut_count']} 次")

            if llm_ok:
                cache = os.path.join(cache_dir(root, "topics", group),
                                     os.path.splitext(name)[0] + suffix + ".json")
                llm_topics(m["segments"], cache, args.llm_model)
                shares = topic_shares(m["segments"])
                if shares:
                    top = ", ".join(f"{t} {pct(s, sum(shares.values())):.0f}%"
                                    for t, s in shares.most_common(3))
                    log(f"    内容类型：{top}")

            videos.append(m)

    if not videos:
        raise SystemExit("没有可分析的视频（检查字幕 / 路径 / --only 过滤）")

    # ---- 汇总 ----
    log("\n" + "=" * 72)
    log("汇总统计 ...")
    groups_videos = {g: [v for v in videos if v["group"] == g] for g in ("manual", "auto")}
    agg = {}
    for g, vids in groups_videos.items():
        if vids:
            agg[g] = aggregate(g, vids)

    cnt_manual = word_counts([e for v in groups_videos["manual"] for e in v["entries"]])
    cnt_auto = word_counts([e for v in groups_videos["auto"] for e in v["entries"]])
    scored = distinctive_words(cnt_auto, cnt_manual,
                              sum(cnt_auto.values()), sum(cnt_manual.values()))

    for g in ("manual", "auto"):
        if g in agg:
            a = agg[g]
            log(f"  {GROUP_LABEL[g]}: {a['n_videos']} 个视频 / {fmt_min(a['duration'])} 分钟｜"
                f"语音密度 {a['speech_ratio']:.1f}%｜主播占语音 "
                f"{a['streamer_ratio_of_speech']:.1f}%｜每 10 分钟 {a['seg_per_10min']:.1f} 段"
                f"（中位 {a['seg_median']:.0f}s）")

    # ---- 大模型总结 ----
    llm_text, video_notes = "", {}
    if llm_ok:
        log("\n大模型总结 ...")
        llm_text = llm_summary(agg, videos, args.llm_model)
        for v in videos:
            note = llm_video_note(v, args.llm_model)
            if note:
                video_notes[v["name"]] = note
            log(f"  - {v['name']}：{'已生成' if note else '跳过'}")

    # ---- 图表 ----
    log("\n出图 ...")
    plt = setup_matplotlib()
    charts = [
        ("01_结构概览.png", "图 1 成片结构概览"),
        ("02_段落时长.png", "图 2 段落时长分布"),
        ("03_风格雷达.png", "图 3 文字风格雷达"),
        ("04_特征词.png", "图 4 特征词"),
        ("05_内容类型.png", "图 5 内容类型分布"),
        ("06_说话时间线.png", "图 6 说话时间线"),
    ]
    made = []
    for fname, title in charts:
        path = os.path.join(root, fname)
        try:
            if fname.startswith("01"):
                chart_overview(videos, path, plt)
            elif fname.startswith("02"):
                chart_segments(videos, path, plt)
            elif fname.startswith("03"):
                chart_style(agg, path, plt)
            elif fname.startswith("04"):
                chart_words(scored, path, plt)
            elif fname.startswith("05"):
                chart_topics(videos, path, plt)
            elif fname.startswith("06"):
                chart_timeline(videos, path, plt)
            if os.path.isfile(path):
                made.append((fname, title))
                log(f"  {fname}")
        except Exception as exc:
            log(f"  ! {fname} 生成失败：{exc}")

    # ---- 数据落盘 ----
    metrics = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "seg_gap": args.seg_gap, "seg_min": args.seg_min, "seg_max": args.seg_max,
            "scene_threshold": args.scene_threshold, "cuts": args.cuts,
            "limit_minutes": args.limit_minutes, "llm": llm_ok,
        },
        "groups": agg,
        "videos": [{k: v for k, v in m.items() if k != "entries"} for m in videos],
    }
    write_json(os.path.join(root, "metrics.json"), metrics)

    csv_path = os.path.join(root, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="\n") as f:
        heads = ["组", "视频", "时长(分)", "语音密度%", "主播占语音%", "主播占视频%",
                 "字幕条数", "语速(字/秒)", "字幕平均字数", "段数", "段长中位(s)",
                 "段长均值(s)", "最长段(s)", "每10分钟段数", "切换次数", "切换/10分钟",
                 "问句率%", "感叹率%", "语气词密度‰", "口头禅密度‰", "互动词密度‰",
                 "用字多样性(Guiraud)"]
        f.write(",".join(heads) + "\n")
        for v in videos:
            st = v["style"]
            row = [GROUP_LABEL[v["group"]], v["name"], f"{v['duration_eff'] / 60:.1f}",
                   f"{v['speech_ratio']:.1f}", f"{v['streamer_ratio_of_speech']:.1f}",
                   f"{v['streamer_ratio_of_video']:.1f}", v["n_subs"],
                   f"{v['chars_per_sec']:.2f}", f"{v['mean_sub_chars']:.1f}",
                   v["seg_count"], f"{v['seg_median']:.0f}", f"{v['seg_mean']:.0f}",
                   f"{v['seg_max']:.0f}", f"{v['seg_per_10min']:.1f}", v["cut_count"],
                   f"{v['cut_per_10min']:.1f}", f"{st['问句率']:.1f}", f"{st['感叹率']:.1f}",
                   f"{st['语气词密度']:.1f}", f"{st['口头禅密度']:.1f}",
                   f"{st['互动词密度']:.1f}", f"{st['用字多样性']:.3f}"]
            f.write(",".join(str(x).replace(",", " ") for x in row) + "\n")

    md = build_markdown(args, videos, agg, scored, llm_text, video_notes, made)
    md_path = os.path.join(root, "summary.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md + "\n")

    log("\n" + "=" * 72)
    log(f"完成：{len(videos)} 个视频，跳过 {len(skipped)} 个，总耗时 {fmt_hms(time.time() - t0)}")
    log(f"  报告   : {md_path}")
    log(f"  数据   : {os.path.join(root, 'metrics.json')}")
    log(f"  表格   : {csv_path}")
    for fname, _ in made:
        log(f"  图表   : {os.path.join(root, fname)}")
    log("=" * 72)


if __name__ == "__main__":
    main()

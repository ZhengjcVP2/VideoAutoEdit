#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensevoice_srt.py — 最长 9 小时的视频：抽音频 -> 本地 SenseVoice -> SRT

流程:
    1. ffprobe 检查时长（默认上限 MAX_HOURS = 9 小时）
    2. ffmpeg 抽成 16k 单声道 16bit wav（默认放临时目录，--keep-wav 可保留）
    3. 按 CHUNK_SEC 分块（默认 10 分钟）：
       fsmn-vad 找出语音段 -> 合并成 <= MAX_UTT_SEC 的句子
    4. SenseVoiceSmall 逐句识别（有显卡自动用 GPU），可选 ct-punc 加标点
    5. 长句再按标点拆成 <= MAX_SUB_CHARS 字符、<= MAX_SUB_SEC 秒的单行短字幕
       （不折行），时间按该句内的语音区间分摊，避开中间的静音
    6. 字幕纠错（默认开启，需要 DEEPSEEK_API_KEY）：每块字幕按 CORRECT_BATCH_LINES
       行送给大模型，修同音字 / 游戏术语 / 人名 / 多余语气词；没配 key 时自动跳过
       （--no-correct 关闭）
    7. 声纹二分类（默认开启）：cam++ 给每条字幕单独算声纹，与主播声纹比对，
       是主播的字幕前加「（主播）」
    8. 输出 SRT，同时写 JSONL 断点记录，长任务中断后可用 --resume 续跑

主播声纹来源:
    - 指定 --streamer-ref 参考样本：wav/mp3 等音频，或者一段「主播在说话」的
      视频（mp4/flv/mkv…，会自动用 ffmpeg 抽出音轨），默认只取前 REF_MAX_SEC 秒；
    - 不指定：自动把句子聚成两类，取「覆盖时间段更多」的那一类
      （主播贯穿整场，游戏过场语音往往集中在某几分钟），
      再用该类句子的时长加权中心作为主播声纹。
    相似度 >= --streamer-threshold 的句子判为主播。

首次运行会自动从 ModelScope 下载模型（约 1GB），需要联网。

用法:
    python sensevoice_srt.py
    python sensevoice_srt.py sample\testvideo_zzz_day1.mp4
    python sensevoice_srt.py video.mp4 -o out.srt
    python sensevoice_srt.py video.mp4 --limit-minutes 3 --keep-wav
    python sensevoice_srt.py video.mp4 --resume
    python sensevoice_srt.py video.mp4 --device cpu --no-punc
    python sensevoice_srt.py video.mp4 --no-correct          # 关闭大模型字幕纠错
    python sensevoice_srt.py video.mp4 --no-streamer-mark
    python sensevoice_srt.py video.mp4 --streamer-ref host.wav
    python sensevoice_srt.py video.mp4 --streamer-ref host.mp4 --ref-sec 60
    python sensevoice_srt.py video.mp4 --streamer-threshold 0.55
"""

import os
import re
import sys
import json
import time
import wave
import shutil
import argparse
import tempfile
import subprocess
import urllib.request
import urllib.error

import numpy as np

# ---------------------------------------------------------------------------
# 输入 / 输出
# ---------------------------------------------------------------------------
VIDEO_PATH = r"sample\testvideo_zzz_day1.mp4"
OUTPUT_DIR = ""                     # 空 = 与视频同目录

# ---------------------------------------------------------------------------
# 识别参数
# ---------------------------------------------------------------------------
MAX_HOURS = 9.0                     # 视频时长上限（小时）
SR = 16000                          # 采样率
CHUNK_SEC = 600.0                   # 每块处理时长（秒）
MAX_UTT_SEC = 15.0                  # 合成一句话的最长时长（秒）
MERGE_GAP_SEC = 0.25                # 相邻语音段间隔小于该值就合并（秒）

ASR_MODEL = "iic/SenseVoiceSmall"   # SenseVoice 模型
VAD_MODEL = "fsmn-vad"              # 静音/语音检测
PUNC_MODEL = "ct-punc"              # 中文标点恢复
DEVICE = "auto"                     # auto / cuda:0 / cpu
LANGUAGE = "zh"                     # zh / en / yue / ja / ko / auto

MIN_SUB_SEC = 0.30                  # 单条字幕最短时长
MAX_SUB_CHARS = 18                  # 单条字幕最多字符数（单行不折行，主播标记另算）
MAX_SUB_SEC = 5.0                   # 单条字幕最长时长（秒）
MIN_SPK_SEC = 0.40                  # 子段太短时用整句声纹兜底

# ---- 声纹二分类（主播 / 游戏）----
MARK_STREAMER = True                # 默认打开：字幕里标注主播说的话
SPK_MODEL = "cam++"                 # 声纹 embedding 模型（FunASR 自带）
STREAMER_LABEL = "（主播）"          # 主播字幕前缀，可改成 "[主播]"
STREAMER_REF = ""                   # 主播参考样本；空 = 自动注册
STREAMER_SIM_THRESHOLD = 0.45       # 与主播声纹的余弦相似度阈值
ENROLL_SEC = 0.0                    # 自动注册用前多少秒；0 = 用全片
COVERAGE_BIN_SEC = 60.0             # 自动注册时统计「覆盖了多少个时间段」
REF_MAX_SEC = 300.0                 # 参考样本最多用多长（秒），0 = 全部

# ---- 字幕纠错（可选，默认开：调用大模型修同音字 / 术语 / 人名）----
CORRECT_ENABLED = True              # 开关：默认打开，--no-correct 关闭
CORRECT_MODEL = "deepseek-chat"     # OpenAI 兼容接口的模型名
CORRECT_BASE_URL = "https://api.deepseek.com/v1"
CORRECT_API_KEY_ENV = "DEEPSEEK_API_KEY"
CORRECT_TEMPERATURE = 0.2
CORRECT_TIMEOUT = 180
CORRECT_MAX_TOKENS = 8192
CORRECT_BATCH_LINES = 40            # 每次送给大模型多少行（也限制输出长度，防截断）

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def fmt_hms(seconds):
    x = int(max(round(float(seconds)), 0))
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def fmt_srt_time(seconds):
    """秒 -> SRT 时间戳 HH:MM:SS,mmm。"""
    ms = max(int(round(float(seconds) * 1000)), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def require_tools():
    missing = [exe for exe in (FFMPEG, FFPROBE) if shutil.which(exe) is None]
    if missing:
        raise SystemExit(
            f"找不到 {', '.join(missing)}，请先安装 FFmpeg 并加入 PATH\n"
            f"    winget install --id Gyan.FFmpeg -e"
        )


def probe_duration(path):
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败：{proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError("无法读取视频时长")


def extract_wav(video, wav_path):
    """抽音轨为 16k 单声道 16bit wav。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video,
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
        wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"抽取音频失败：{proc.stderr.strip()}")


def load_ref_audio(path, max_sec=REF_MAX_SEC):
    """把主播参考样本解码成 16k 单声道 int16 采样。

    音频（wav/mp3/m4a…）和视频（mp4/flv/mkv…）都可以：视频会先用 ffmpeg
    抽出音轨，所以 --streamer-ref 直接给一段「主播在说话」的视频片段也行。
    """
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-sn", "-dn"]
    if max_sec and max_sec > 0:
        cmd += ["-t", f"{float(max_sec):.3f}"]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return np.frombuffer(proc.stdout, dtype=np.int16)


def iter_wav_chunks(wav_path, chunk_sec=CHUNK_SEC):
    """按块读取 wav，返回 (块序号, 块起点秒, int16 采样)。"""
    with wave.open(wav_path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise RuntimeError("wav 必须是 16bit 单声道")
        if w.getframerate() != SR:
            raise RuntimeError(f"wav 采样率应为 {SR}，实际 {w.getframerate()}")

        frames_per_chunk = int(round(chunk_sec * SR))
        idx = 0
        while True:
            data = w.readframes(frames_per_chunk)
            if not data:
                break
            start_sec = idx * frames_per_chunk / SR
            yield idx, start_sec, np.frombuffer(data, dtype=np.int16)
            idx += 1


# ---------------------------------------------------------------------------
# 语音段处理
# ---------------------------------------------------------------------------
def detect_speech(vad_model, samples, chunk_id):
    """用 fsmn-vad 找语音段，返回 [(start_ms, end_ms), ...]（块内相对时间）。"""
    audio = np.asarray(samples, dtype=np.float32) / 32768.0
    try:
        res = vad_model.generate(input=audio, fs=SR, disable_pbar=True)
    except Exception as exc:
        log(f"  ! 块 {chunk_id} VAD 失败：{exc}")
        return []

    if not res:
        return []
    item = res[0]
    value = item.get("value", []) if isinstance(item, dict) else []

    segments = []
    for v in value:
        try:
            if isinstance(v, dict):
                s, e = v.get("start"), v.get("end")
            else:
                s, e = v[0], v[1]
            segments.append((float(s), float(e)))
        except (TypeError, ValueError, IndexError):
            continue
    return segments


def group_utterances(segments, max_utt_ms=MAX_UTT_SEC * 1000,
                     max_gap_ms=MERGE_GAP_SEC * 1000):
    """把 VAD 语音段合并成句子。

    返回 [(start_ms, end_ms, [(s_ms, e_ms), ...]), ...]；第三个元素是这句里
    真正的语音区间（不含中间的静音），分摊字幕时间时用它避开静音。
    """
    merged = []
    current = None
    for s, e in segments:
        if e <= s:
            continue
        if current is None:
            current = [s, e, [(s, e)]]
            continue
        if s - current[1] <= max_gap_ms and e - current[0] <= max_utt_ms:
            current[1] = e
            current[2].append((s, e))
        else:
            merged.append(current)
            current = [s, e, [(s, e)]]
    if current is not None:
        merged.append(current)

    # 单个语音段过长时，按最大时长硬切
    result = []
    for s, e, spans in merged:
        while e - s > max_utt_ms:
            cut = s + max_utt_ms
            result.append((s, cut, _clip_spans(spans, s, cut)))
            s = cut
        if e > s:
            result.append((s, e, _clip_spans(spans, s, e)))
    return result


def _clip_spans(spans, start, end):
    """把语音区间裁到 [start, end] 之内。"""
    out = []
    for a, b in spans:
        a, b = max(a, start), min(b, end)
        if b > a:
            out.append((a, b))
    return out


_TAG_RE = re.compile(r"<\s*\|\s*[^|<>]*?\s*\|\s*>|<unk>")

# 有没有实际内容（中文 / 数字 / 字母），用来丢掉只剩标点的碎片
_CONTENT_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]")

# 连续出现的标点（中英文都算）
_PUNCT_RUN_RE = re.compile(r"[，。、；：！？…,\.!\?;:]+")
_LEAD_PUNCT_RE = re.compile(r"^[，、。；：！？,;:!?]+")
_CLAUSE_RE = re.compile(r"[^，。！？；：、,.!?;:…]+[，。！？；：、,.!?;:…]*")
_STRONG_PUNCT = "。！？…"

# 没有标点可断时优先切在这些语气词后面，尽量不切断词组
_SOFT_BREAK_AFTER = set("的了着过啊吧呢嘛呀哦呗啦哈")


def _pick_punct(run):
    """一段连续标点只留最强的那个：？ > ！ > 。 > … > ，。"""
    if len(run) <= 1:
        return run                      # 单个标点原样保留（别动 4.7 这种小数点）
    if "？" in run or "?" in run:
        return "？"
    if "！" in run or "!" in run:
        return "！"
    if "。" in run:
        return "。"
    if "…" in run or set(run) <= {".", "．"}:
        return "…"
    if set(run) <= {"；", "：", ";", ":"}:
        return "；" if ("；" in run or ";" in run) else "："
    return "，"


def normalize_punctuation(text):
    """整理标点：合并「，。」「。？」这类相邻标点，去掉句首多余的停顿标点。"""
    if not text:
        return ""
    text = _PUNCT_RUN_RE.sub(lambda m: _pick_punct(m.group(0)), text)
    text = _LEAD_PUNCT_RE.sub("", text)
    return text.strip()


def clean_text(text):
    """去掉 SenseVoice 的语种/情感/事件标签，并规范标点。"""
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    text = " ".join(text.split())
    return normalize_punctuation(text)


def _hard_split(clause, max_chars):
    """一整个小句太长（中间没标点）时，退而切在「的 / 了 / 啊 …」后面。"""
    pieces, rest = [], clause
    low = max(max_chars // 2, 1)
    while len(rest) > max_chars:
        cut = max_chars
        for i in range(max_chars, low - 1, -1):
            if rest[i - 1] in _SOFT_BREAK_AFTER:
                cut = i
                break
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _char_time(pos, total_chars, start, end, spans):
    """把「第几个字」换算成绝对时间；有语音区间时只按说话时长推进。"""
    frac = min(max(pos / max(total_chars, 1), 0.0), 1.0)
    if not spans:
        return start + (end - start) * frac
    total = sum(b - a for a, b in spans)
    target, acc = total * frac, 0.0
    for a, b in spans:
        if acc + (b - a) >= target - 1e-9:
            return min(max(a + (target - acc), a), b)
        acc += b - a
    return spans[-1][1]


def split_subtitles(text, start, end, max_chars=MAX_SUB_CHARS,
                    max_sec=MAX_SUB_SEC, speech=None):
    """把一句识别结果拆成多条「单行、不换行」的短字幕。

    - 只在标点处断句，断句位置尽量落在语义边界上（实在没标点才用软断点）；
    - 每条不超过 max_chars 个字符、max_sec 秒；
    - 给了 speech（这句里真正的语音区间，秒）时按语音时长分摊时间，
      中间的静音不计入字数，边界更贴近真实说话位置。
    返回 [(start, end, text), ...]。
    """
    text = normalize_punctuation((text or "").strip())
    if not text or not _CONTENT_RE.search(text):
        return []

    spans = []
    for a, b in (speech or []):
        a, b = max(a, start), min(b, end)
        if b > a:
            spans.append((a, b))
    voiced = sum(b - a for a, b in spans) if spans else max(end - start, 0.0)

    units = [u.strip() for u in _CLAUSE_RE.findall(text) if u.strip()]
    if not units:
        units = [text]

    pieces = []
    for unit in units:
        if not _CONTENT_RE.search(unit):
            continue                    # 只剩标点的碎片直接丢掉
        pieces.extend(_hard_split(unit, max_chars))
    if not pieces:
        return []

    total_chars = sum(len(p) for p in pieces) or 1

    groups = []
    current, current_chars, consumed = "", 0, 0
    for piece in pieces:
        add = len(piece)
        end_pos = consumed + current_chars + add
        too_long = bool(current) and (
            current_chars + add > max_chars
            or voiced * end_pos / total_chars > max_sec)
        # 已经攒了半行、又刚好说完一句话 -> 也断开，提高断句频率
        sentence_done = (bool(current) and current_chars >= max_chars / 2
                         and current[-1] in _STRONG_PUNCT)
        if bool(current) and (too_long or sentence_done):
            groups.append((consumed, consumed + current_chars, current))
            consumed += current_chars
            current, current_chars = "", 0
        current += piece
        current_chars += add
    if current:
        groups.append((consumed, consumed + current_chars, current))

    result = []
    for char_start, char_end, group_text in groups:
        group_text = group_text.strip()
        if not group_text or not _CONTENT_RE.search(group_text):
            continue
        s = _char_time(char_start, total_chars, start, end, spans)
        e = _char_time(char_end, total_chars, start, end, spans)
        if e <= s:
            e = s + MIN_SUB_SEC
        result.append((s, e, group_text))
    return result


# ---------------------------------------------------------------------------
# SRT / JSONL
# ---------------------------------------------------------------------------
def write_srt(segments, path, streamer_label=""):
    """segments: [{"start": 秒, "end": 秒, "text": str, "is_streamer": bool}, ...]

    每条字幕都是单行、不折行：太长的话在这里再拆一次，主播标记挂在每一条
    的开头。
    """
    entries = []
    for rec in sorted(segments, key=lambda r: (r["start"], r["end"])):
        start = float(rec["start"])
        end = float(rec["end"])
        if end <= start:
            continue
        label = streamer_label if (streamer_label and rec.get("is_streamer")) else ""
        for sub_start, sub_end, sub_text in split_subtitles(rec.get("text", ""),
                                                            start, end):
            entries.append([sub_start, sub_end, label + sub_text])

    # 保证每条都能看清，又不压到下一条
    for i, entry in enumerate(entries):
        start, end = entry[0], entry[1]
        if end - start < MIN_SUB_SEC:
            end = start + MIN_SUB_SEC
        if i + 1 < len(entries):
            nxt = entries[i + 1][0]
            if end > nxt:
                end = nxt if nxt - start >= 0.05 else start + 0.05
        entry[1] = end

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f"{i}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n"
                    f"{text}\n\n")
    return len(entries)


def read_checkpoint(path):
    """读取 JSONL 断点：返回 (句子列表, 已完成的块序号集合)。"""
    segments, done = [], set()
    if not os.path.isfile(path):
        return segments, done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("done"):
                done.add(rec.get("chunk"))
            elif "text" in rec:
                segments.append(rec)
    return segments, done


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_models(device, use_punc, use_spk):
    from funasr import AutoModel

    common = dict(device=device, disable_update=True,
                  disable_pbar=True, disable_log=True)
    log("加载 VAD 模型 ...")
    vad_model = AutoModel(model=VAD_MODEL, **common)

    log("加载 SenseVoice 模型 ...")
    asr_model, has_punc = None, False
    if use_punc:
        try:
            asr_model = AutoModel(model=ASR_MODEL,
                                  punc_model=PUNC_MODEL, **common)
            has_punc = True
        except Exception as exc:
            log(f"标点模型加载失败，改用无标点模式：{exc}")
    if asr_model is None:
        asr_model = AutoModel(model=ASR_MODEL, **common)

    spk_model = None
    if use_spk:
        log("加载声纹模型 ...")
        try:
            spk_model = AutoModel(model=SPK_MODEL, **common)
        except Exception as exc:
            log(f"声纹模型加载失败，关闭主播标注：{exc}")

    return vad_model, asr_model, spk_model, has_punc


def resolve_device(device):
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def transcribe_utterance(asr_model, samples, language):
    audio = np.asarray(samples, dtype=np.float32) / 32768.0
    if audio.size == 0:
        return ""
    try:
        res = asr_model.generate(
            input=audio, fs=SR, language=language,
            use_itn=True, ban_emo_unk=True, disable_pbar=True)
    except Exception as exc:
        log(f"  ! 识别失败（{audio.size / SR:.1f}s）：{exc}")
        return ""
    if not res:
        return ""
    item = res[0]
    text = item.get("text", "") if isinstance(item, dict) else str(item)
    return clean_text(text)


# ---------------------------------------------------------------------------
# 声纹二分类：主播 / 非主播
# ---------------------------------------------------------------------------
def speaker_embedding(spk_model, audio):
    """audio 为 int16 ndarray 或音频路径；返回单位化 embedding，失败返回 None。"""
    try:
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32) / 32768.0
        res = spk_model.generate(input=audio, fs=SR, disable_pbar=True)
    except Exception as exc:
        log(f"  ! 声纹提取失败：{exc}")
        return None

    if not res:
        return None
    item = res[0]
    emb = item.get("spk_embedding") if isinstance(item, dict) else None
    if emb is None:
        return None
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 1e-8 else None


def _kmeans2(x, iters=40):
    """把单位化 embedding 简单分成 2 类，返回 (labels, center0, center1)。"""
    c0 = x[0].copy()
    c1 = x[int(np.argmin(x @ c0))].copy()
    labels = np.full(len(x), -1, dtype=np.int8)

    for _ in range(iters):
        new_labels = (x @ c1 > x @ c0).astype(np.int8)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            mask = labels == k
            if not mask.any():
                continue
            center = x[mask].mean(axis=0)
            norm = float(np.linalg.norm(center))
            if norm < 1e-8:
                continue
            center = center / norm
            if k == 0:
                c0 = center
            else:
                c1 = center
    return labels, c0, c1


def build_streamer_reference(records, spk_model, ref_audio, enroll_sec,
                             ref_sec=REF_MAX_SEC):
    """返回 (单位化主播声纹, 说明)；失败返回 (None, 原因)。

    给了 ref_audio 就直接拿它算参考声纹（音频 / 视频都行，视频先抽音轨）；
    没给、读不出来或样本太短时，退回自动注册。
    自动注册的思路：把所有句子聚成 2 类，主播通常「贯穿整场」，
    所以取覆盖时间段更多的那一类（相同时取说话时长更多的），
    而不是单纯看哪类说得久——游戏过场语音可能集中在某几分钟里。
    """
    if ref_audio:
        if os.path.isfile(ref_audio):
            samples = None
            try:
                samples = load_ref_audio(ref_audio, ref_sec)
            except Exception as exc:
                log(f"  ! 读取主播参考样本失败（{os.path.basename(ref_audio)}）："
                    f"{exc}，改用自动注册")
            if samples is not None and samples.size >= SR * 0.5:
                emb = speaker_embedding(spk_model, samples)
                if emb is not None:
                    return (emb, f"参考样本 {os.path.basename(ref_audio)}"
                                 f"（{samples.size / SR:.0f}s）")
                log("  ! 参考样本提取声纹失败，改用自动注册")
            elif samples is not None:
                log(f"  ! 参考样本太短（{samples.size / SR:.1f}s），改用自动注册")
        else:
            log(f"  ! 找不到主播参考样本：{ref_audio}，改用自动注册")

    pool = [r for r in records if r.get("emb")]
    if enroll_sec and enroll_sec > 0:
        pool = [r for r in pool if r.get("start", 0) < enroll_sec]
    if len(pool) < 6:
        return None, f"可用的句子只有 {len(pool)} 句，无法自动注册"

    x = np.asarray([r["emb"] for r in pool], dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    labels, c0, c1 = _kmeans2(x)

    stats = []
    for k in (0, 1):
        members = [r for r, lb in zip(pool, labels) if lb == k]
        if not members:
            stats.append((0, 0.0, 0))
            continue
        coverage = len({int(r["start"] // COVERAGE_BIN_SEC) for r in members})
        duration = sum(r["end"] - r["start"] for r in members)
        stats.append((coverage, duration, len(members)))

    # 覆盖时间段更多 = 更像贯穿全场的主播；其次比总时长
    if (stats[0][0], stats[0][1]) >= (stats[1][0], stats[1][1]):
        picked = 0
    else:
        picked = 1

    members = [r for r, lb in zip(pool, labels) if lb == picked]
    if len(members) < 3:
        return None, "自动注册的句子太少"

    weights = np.asarray(
        [max(r["end"] - r["start"], 0.1) for r in members], dtype=np.float32)
    center = (x[labels == picked] * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(center))
    if norm < 1e-8:
        return None, "自动注册的中心向量无效"
    center = center / norm

    note = (f"自动注册（类{picked}：覆盖 {stats[picked][0]} 段时间 / "
            f"{stats[picked][2]} 句 / {stats[picked][1]:.0f}s）")
    return center, note


def classify_streamer(records, spk_model, ref_audio,
                      threshold=STREAMER_SIM_THRESHOLD, enroll_sec=ENROLL_SEC,
                      ref_sec=REF_MAX_SEC):
    """给 records 打 is_streamer 标记，返回 (命中数, 说明)。"""
    usable = [r for r in records if r.get("emb")]
    if not usable:
        return 0, "没有声纹数据（旧断点文件？）"

    ref, note = build_streamer_reference(
        records, spk_model, ref_audio, enroll_sec, ref_sec)
    if ref is None:
        return 0, note

    hit = 0
    for rec in usable:
        emb = np.asarray(rec["emb"], dtype=np.float32)
        sim = float(np.dot(emb, ref))
        rec["is_streamer"] = sim >= threshold

    # 同一句话拆出来的几条字幕标记保持一致：短句声纹不稳，只要有一条命中主播
    # 就整句都算主播，保证每一句开头都有标记
    groups = {}
    for rec in usable:
        utt = rec.get("utt")
        if utt is None:
            continue
        groups.setdefault((rec.get("chunk"), utt), []).append(rec)
    for members in groups.values():
        if any(r["is_streamer"] for r in members):
            for r in members:
                r["is_streamer"] = True

    hit = sum(1 for rec in usable if rec["is_streamer"])
    return hit, f"{note}，阈值 {threshold:g}"


# ---------------------------------------------------------------------------
# 字幕纠错（可选，调用大模型：修同音字 / 术语 / 人名 / 多余语气词）
# ---------------------------------------------------------------------------
CORRECT_SYSTEM = """你是中文游戏直播字幕（ASR）的校对员。输入是带行号的语音识别字幕，
每行是一条很短的屏幕上字幕，常常只有半句话，多行连起来才是完整句子。
常见问题：同音字错认、游戏术语 / 角色名 / 技能名 / 人名写错、重复的语气词。

要求：
- 只输出 JSON：{{"lines": [{{"i": 行号, "text": "改好的文本"}}]}}
- 只列确实需要修改的行，不需要改的行不要出现；最多输出 {max_lines} 行；
- 不要合并、拆分或新增条目，行号必须与输入一一对应；
- 结合前后行理解意思再改；不要脑补原文没有的信息，拿不准就保持原样；
- 不要添加「（主播）」之类的说话人标记；
- 改完每行的长度要和原文接近（这是字幕，太长会超出屏幕）。
"""


def llm_api_key():
    """大模型 key：环境变量优先，运行时再读（方便先 set 再调用）。"""
    return os.environ.get(CORRECT_API_KEY_ENV, "").strip()


def llm_chat(system_prompt, user_prompt, model=CORRECT_MODEL):
    """调 OpenAI 兼容接口，返回文本回复。"""
    key = llm_api_key()
    if not key:
        raise RuntimeError(f"未设置环境变量 {CORRECT_API_KEY_ENV}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": CORRECT_TEMPERATURE,
        "max_tokens": CORRECT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        CORRECT_BASE_URL.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=CORRECT_TIMEOUT) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:300]}")
    except Exception as exc:
        raise RuntimeError(f"大模型调用失败: {exc}")

    return obj["choices"][0]["message"]["content"]


def parse_json_object(text):
    """从可能带 ``` 围栏的回答里取出 JSON 对象。"""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i:j + 1])
    raise ValueError("大模型返回的内容不是合法 JSON")


def correct_records(records, batch_lines=CORRECT_BATCH_LINES,
                    model=CORRECT_MODEL):
    """用大模型逐条纠错，原地修改 records[i]["text"]；返回修正条数。

    每 batch_lines 行发一次请求（行数少，输出不容易被截断）；
    任何一次失败都只打日志、保留原文，不影响 ASR 主流程。
    """
    fixed = 0
    for i in range(0, len(records), batch_lines):
        batch = records[i:i + batch_lines]
        rows = [f"{j}\t{rec.get('text', '')}" for j, rec in enumerate(batch)]
        lo = float(batch[0].get("start", 0.0))
        hi = float(batch[-1].get("end", 0.0))
        system = CORRECT_SYSTEM.format(max_lines=len(batch))
        user = (f"本段范围：{lo:.1f}s - {hi:.1f}s（共 {len(batch)} 行）\n\n"
                "字幕：\n" + "\n".join(rows))

        try:
            obj = parse_json_object(llm_chat(system, user, model=model))
        except Exception as exc:
            log(f"  ! 字幕纠错失败（{fmt_hms(lo)} 起 {len(batch)} 行），"
                f"保留原文：{exc}")
            continue

        lines = obj.get("lines", []) if isinstance(obj, dict) else []
        for item in lines if isinstance(lines, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                j = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            if not (0 <= j < len(batch)):
                continue

            text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            if text.startswith(STREAMER_LABEL):
                text = text[len(STREAMER_LABEL):].strip()
            text = normalize_punctuation(text)
            if not text or text == batch[j].get("text"):
                continue
            batch[j]["text"] = text
            fixed += 1
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="最长 9 小时视频：抽音频 + 本地 SenseVoice -> SRT")
    parser.add_argument("video", nargs="?", default=VIDEO_PATH, help="输入视频")
    parser.add_argument("-o", "--output", default=None, help="输出 SRT 路径")
    parser.add_argument("--jsonl", default=None, help="断点 JSONL 路径")
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS,
                        help=f"视频时长上限（小时），默认 {MAX_HOURS:g}")
    parser.add_argument("--chunk", type=float, default=CHUNK_SEC,
                        help=f"每块处理时长（秒），默认 {CHUNK_SEC:g}")
    parser.add_argument("--limit-minutes", type=float, default=0,
                        help="只处理前 N 分钟（调试用，0 = 全部）")
    parser.add_argument("--device", default=DEVICE, help="auto / cuda:0 / cpu")
    parser.add_argument("--language", default=LANGUAGE,
                        help="zh / en / yue / ja / ko / auto")
    parser.add_argument("--resume", action="store_true",
                        help="从 JSONL 断点继续")
    parser.add_argument("--keep-wav", action="store_true",
                        help="保留抽取出来的 wav")
    parser.add_argument("--no-punc", action="store_true",
                        help="不加标点模型（更快）")
    parser.add_argument("--no-correct", dest="correct", action="store_false",
                        default=CORRECT_ENABLED,
                        help=f"关闭字幕纠错（默认为开，需要环境变量 "
                             f"{CORRECT_API_KEY_ENV}）")
    parser.add_argument("--correct-model", default=CORRECT_MODEL,
                        help=f"字幕纠错用的大模型，默认 {CORRECT_MODEL}")
    parser.add_argument("--no-streamer-mark", action="store_true",
                        help="关闭主播声纹标注")
    parser.add_argument("--streamer-ref", default=STREAMER_REF,
                        help="主播参考样本：wav/mp3 音频，或一段主播说话的视频（mp4/flv…）；"
                             "不给则自动注册")
    parser.add_argument("--streamer-threshold", type=float,
                        default=STREAMER_SIM_THRESHOLD,
                        help=f"主播声纹相似度阈值，默认 {STREAMER_SIM_THRESHOLD:g}")
    parser.add_argument("--ref-sec", type=float, default=REF_MAX_SEC,
                        help=f"参考样本最多用前多少秒，0 = 全部，默认 {REF_MAX_SEC:g}")
    parser.add_argument("--enroll-sec", type=float, default=ENROLL_SEC,
                        help=f"自动注册只用开头多少秒，0 = 用全片（默认 {ENROLL_SEC:g}）")
    args = parser.parse_args()

    mark_streamer = MARK_STREAMER and not args.no_streamer_mark

    do_correct = bool(args.correct)
    if do_correct and not llm_api_key():
        log(f"  ! 未设置环境变量 {CORRECT_API_KEY_ENV}，跳过字幕纠错")
        log(f"    （PowerShell: $env:{CORRECT_API_KEY_ENV}=\"sk-xxxx\"）")
        do_correct = False

    require_tools()

    video = args.video
    if not os.path.isfile(video):
        raise SystemExit(f"找不到视频文件: {video}")

    duration = probe_duration(video)
    limit_sec = duration
    if args.limit_minutes > 0:
        limit_sec = min(duration, args.limit_minutes * 60.0)

    if limit_sec > args.max_hours * 3600 + 1:
        raise SystemExit(
            f"视频时长 {fmt_hms(duration)} 超过上限 {args.max_hours:g} 小时")

    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    srt_path = args.output or os.path.join(out_dir, stem + ".srt")
    jsonl_path = args.jsonl or os.path.join(out_dir, stem + ".asr.jsonl")

    if not args.resume and os.path.isfile(jsonl_path):
        os.remove(jsonl_path)

    device = resolve_device(args.device)
    log(f"视频     : {video}")
    log(f"时长     : {fmt_hms(duration)}（本次处理 {fmt_hms(limit_sec)}）")
    log(f"设备     : {device}")
    log(f"输出     : {srt_path}")
    log(f"字幕纠错 : {'开（' + args.correct_model + '）' if do_correct else '关'}")
    if mark_streamer:
        ref_note = (f"参考 {os.path.basename(args.streamer_ref)}"
                    if args.streamer_ref else "自动注册")
        log(f"主播标注 : 开（{ref_note}，阈值 {args.streamer_threshold:g}）")
    else:
        log("主播标注 : 关")

    # ---- 抽音频 ----
    tmp_dir = tempfile.mkdtemp(prefix="sensevoice_")
    wav_path = os.path.join(tmp_dir, "audio.wav")
    all_records, spk_model = [], None
    fixed_total = 0
    try:
        log("抽取音频 ...")
        t0 = time.time()
        extract_wav(video, wav_path)
        log(f"音频完成 : {os.path.getsize(wav_path) / 2**20:.1f} MB，"
            f"耗时 {time.time() - t0:.1f}s")

        # ---- 加载模型 ----
        vad_model, asr_model, spk_model, has_punc = build_models(
            device, use_punc=not args.no_punc, use_spk=mark_streamer)
        if spk_model is None:
            mark_streamer = False
        log(f"模型就绪 : SenseVoice{'（含标点）' if has_punc else ''}"
            f"{' + 声纹' if spk_model is not None else ''}")

        # ---- 断点 ----
        all_records, done_chunks = read_checkpoint(jsonl_path)
        if args.resume:
            all_records = [r for r in all_records if r.get("chunk") in done_chunks]
            log(f"断点续跑 : 已完成 {len(done_chunks)} 块，"
                f"已有 {len(all_records)} 句")
        else:
            all_records, done_chunks = [], set()

        total_chunks = max(int(np.ceil(limit_sec / args.chunk)), 1)
        log(f"分块识别 : 共 {total_chunks} 块，每块 {args.chunk:g}s\n")

        started = time.time()
        for chunk_id, chunk_start, samples in iter_wav_chunks(wav_path, args.chunk):
            if chunk_start >= limit_sec:
                break
            if chunk_id in done_chunks:
                continue

            pct = min(chunk_start / max(limit_sec, 1e-9), 1.0) * 100
            elapsed = time.time() - started
            eta = elapsed / max(pct, 0.1) * (100 - pct) / 100 if pct > 0 else 0
            log(f"[{chunk_id + 1}/{total_chunks}] {fmt_hms(chunk_start)} "
                f"({pct:4.1f}%)  已用 {fmt_hms(elapsed)}  "
                f"预计剩余 {fmt_hms(eta)}")

            vad_segs = detect_speech(vad_model, samples, chunk_id)
            utterances = group_utterances(vad_segs)

            records = []
            for utt_id, (s_ms, e_ms, spans) in enumerate(utterances):
                start = chunk_start + s_ms / 1000.0
                end = min(chunk_start + e_ms / 1000.0, limit_sec)
                if end - start < MIN_SUB_SEC:
                    continue

                i0 = int(round((start - chunk_start) * SR))
                i1 = int(round((end - chunk_start) * SR))
                piece = samples[i0:i1]
                if piece.size < SR * 0.15:
                    continue

                text = transcribe_utterance(asr_model, piece, args.language)
                if not text:
                    continue

                # 长句拆成多条单行短字幕，每条单独算声纹；时间按语音区间分摊
                speech = [(chunk_start + a / 1000.0, chunk_start + b / 1000.0)
                          for a, b in spans]
                subs = split_subtitles(text, start, end, speech=speech)
                parent_emb = None
                for sub_start, sub_end, sub_text in subs:
                    sub_end = min(sub_end, end)
                    if sub_end <= sub_start:
                        continue

                    record = {"chunk": chunk_id, "utt": utt_id,
                              "start": sub_start, "end": sub_end,
                              "text": sub_text}
                    if spk_model is not None:
                        j0 = int(round((sub_start - chunk_start) * SR))
                        j1 = int(round((sub_end - chunk_start) * SR))
                        seg = samples[j0:j1]
                        if seg.size >= int(MIN_SPK_SEC * SR):
                            emb = speaker_embedding(spk_model, seg)
                        else:
                            if parent_emb is None:
                                parent_emb = speaker_embedding(spk_model, piece)
                            emb = parent_emb
                        if emb is not None:
                            record["emb"] = [round(float(v), 6) for v in emb]
                    records.append(record)

            # 字幕纠错：改写字幕文本，写进断点后就不会重复纠错了
            fixed = 0
            if do_correct and records:
                fixed = correct_records(records, model=args.correct_model)
                fixed_total += fixed

            append_jsonl(jsonl_path, records)
            append_jsonl(jsonl_path, [{"chunk": chunk_id, "done": True}])
            all_records.extend(records)
            log(f"    -> 语音段 {len(utterances)}，有效句子 {len(records)}"
                + (f"，纠错 {fixed} 条" if do_correct else ""))

    except KeyboardInterrupt:
        log("\n收到中断，先把已有结果写成 SRT ...")
    finally:
        if mark_streamer and spk_model is not None and all_records:
            hit, note = classify_streamer(
                all_records, spk_model, args.streamer_ref,
                args.streamer_threshold, args.enroll_sec, args.ref_sec)
            log(f"主播标注 : {hit}/{len(all_records)} 句（{note}）")

        if do_correct:
            log(f"字幕纠错 : 共修正 {fixed_total} 条")

        count = write_srt(all_records, srt_path,
                          STREAMER_LABEL if mark_streamer else "")
        log(f"\n已写入 {count} 条字幕: {srt_path}")
        log(f"断点文件: {jsonl_path}")
        if args.keep_wav:
            log(f"音频保留: {wav_path}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

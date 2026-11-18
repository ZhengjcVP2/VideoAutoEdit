#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edit_with_caption.py — 用字幕（可选叠加弹幕）交给大模型挑选精华片段并顺序剪辑

思路:
    1. 读 SRT（可能含「（主播）」等标注，标注不一定可靠，只当提示）
    2. 可选读弹幕 XML：弹幕是真实时间，而视频每 30 分钟会重复 4 秒，
       所以弹幕时间要「后移」才能对齐视频时间，再按 60s 聚合成热度桶；
       聚合时过滤掉弹幕表情，只看真正的文字，并标出明显高于均值的峰值桶
    3. 字幕按 LLM_CHUNK_SEC 分块（默认 20 分钟），逐块交给大模型分析
       （字幕纠错已经在字幕阶段由 sensevoice_srt.py 做过，这里直接拿来用）
    4. 逐块请大模型：结合前后文「上下文」行和弹幕热度，给出候选精彩片段
       （起止时间必须取自字幕边界、落在本块范围内）
    5. 汇总候选，再请大模型带着候选附近的字幕上下文做一次全局挑选
    5.5 内容按 4 类管：剧情主线 / 闲聊杂谈 / 吐槽玩梗 / 其他（其余全部并成「其他」，
       样本少的类型单独统计没意义）；四类的时长比例不是写死的，而是按候选池
       实际供给 + 每类上下限（如主线 20%~60%）动态分配
    5.6 逻辑连接：主播吐槽 / 评论某条信息时，如果那条信息就在上文很近的地方
       （关键词接续 / 指代上一句），把出处一起并进片段；关联不强的上下文不补，
       保证内容完整的同时时长尽量紧凑
    6. 程序做硬校验：时间顺序、互不重叠、单段 >= MIN_CLIP_SEC、
       总时长落在 [TARGET_MIN_SEC, TARGET_MAX_SEC]
    7. 音频对齐：每个剪辑点向外扩最多 SNAP_MAX_SHIFT 秒，切到附近最安静的位置，
       避免把句子或剧情切一半（与 auto_edit.py 的做法一致）；
       因此带来的轻微超长（默认 OVERFLOW_TOLERANCE_SEC 内）优先保片段完整
    8. 用 stream copy 逐段切 + concat 拼接（剪辑逻辑与 auto_edit.py 一致），
       最后再整体重编码一遍成恒定帧率（CFR）：stream copy 拼出来的时间戳是
       「拼接式」的，Premiere 会按平均帧率推算导致时间错位，重编码后即对齐
       （码率上限沿用源片，画质 / 体积基本不变）
       **分片编码不一致时**（典型：一部分 HEVC、一部分 H.264）不能 stream copy，
       会把两种码流写进同一条轨道，解码器从异构段开始整段解不出来（画面可能
       冻结几十秒到十几分钟），这时自动改为「逐段重编码成统一 H.264/AAC 再拼接」

为什么用大模型而不是纯弹幕密度:
    弹幕在「主播回答问题」时最密集，纯密度会把成片剪成问答；
    大模型可以同时看观众反应、主播发言和游戏内容，按用户给的偏好挑重点。

组合剪辑（多个视频 = 同一个视频被切成的若干分段）:
    加 --video-dir <文件夹>，文件夹里的视频按文件名（字母顺序，数字按大小，
    即 part2 在 part10 之前）拼成一条完整时间轴，每段的时长累加得到全局偏移。
    字幕 / 弹幕的取法：
      - 传文件夹（--srt / --danmaku 指向目录，默认与 --video-dir 相同）：
        按分片一一配对（优先同名，其次按字母顺序）；SRT 的时间是相对各分片的，
        会加上该分片的全局偏移；弹幕时间也视为相对本分片；
      - 传单个文件：视为已经覆盖全部视频的「全局」字幕 / 弹幕，时间直接使用。
    剪辑时每个片段会按需跨分片拆分，从各自的分片里切出来再依次拼接，
    所以成片可以横跨多个源文件，而挑选 / 时长校验仍在全局时间轴上做。

输出位置:
    成片、剪辑计划（.plan.json / .plan.txt）和成片字幕（.srt）默认都写到
    <输入文件夹>/export/ 里；分片自己的 ASR 字幕仍然留在视频旁边（配对用）。
    用 -o / --srt-out 可以另外指定。

用法:
    python edit_with_caption.py
    python edit_with_caption.py --start 0 --end 3600
    python edit_with_caption.py --prompt "主播经常漏看东西，重点找眼神不好的吐槽"
    python edit_with_caption.py --use-danmaku
    python edit_with_caption.py --video-dir video_source\\xxx --use-danmaku
                                                   # 组合剪辑文件夹里的多个分段
    python edit_with_caption.py --no-snap          # 关闭剪辑点音频对齐（不向外扩边界）
    python edit_with_caption.py --dry-run          # 只输出剪辑计划，不渲染
    set DEEPSEEK_API_KEY=sk-xxxx                   # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-xxxx"
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import tempfile
import bisect
import subprocess
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from auto_edit import (
    FFMPEG, FFPROBE, fmt_hms, require_tools, probe_duration,
    cut_segment, concat_segments, snap_cut,
)

# ---------------------------------------------------------------------------
# 输入 / 输出
# ---------------------------------------------------------------------------
VIDEO_PATH = r"sample\testvideo_zzz_day1.mp4"
SRT_PATH = r"sample\testvideo_zzz_day1.srt"
DANMAKU_PATH = r"sample\test1.xml"
VIDEO_DIR = r"video_source\文七星铁3测day1-2"  # 组合剪辑：文件夹内多个视频按字母顺序当成分段
OUTPUT_PATH = ""                    # 空 = <源视频文件夹>\export\<名字>_edited.mp4

# 成片 / 剪辑计划 / 成片字幕统一放进输入文件夹下的这个子文件夹
# （分片自己的 ASR 字幕仍然留在视频旁边，供配对使用）
EXPORT_DIR_NAME = "export"

# 组合剪辑时识别哪些文件
VIDEO_EXTS = {".mp4", ".flv", ".mkv", ".ts", ".m4v", ".webm", ".mov", ".avi"}
SRT_EXTS = {".srt"}
DANMAKU_EXTS = {".xml"}

# ---------------------------------------------------------------------------
# 剪辑参数
# ---------------------------------------------------------------------------
VIDEO_START = 0.0                   # 只使用视频的这个时间点之后（秒）
VIDEO_END = None                    # 只使用到这个时间点（秒），None = 视频结尾
TARGET_MIN_SEC = 30 * 60            # 成片最短时长（秒）
TARGET_MAX_SEC = 40 * 60            # 成片最长时长（秒）
MIN_CLIP_SEC = 15.0                 # 每个片段的时间下限（秒）
EDIT_PROMPT = ""                    # 剪辑方向 / 提示词，例如「眼神」「剧情高光」

# ---------------------------------------------------------------------------
# 弹幕参数
# ---------------------------------------------------------------------------
USE_DANMAKU = False                 # 开关：是否把弹幕作为辅助信号交给大模型
DUP_INTERVAL = 30 * 60              # 视频每 30 分钟重复一次
DUP_LEN = 4.0                       # 每次重复 4 秒
DANMAKU_BUCKET_SEC = 60.0           # 弹幕热度聚合桶宽（秒）
DANMAKU_SAMPLES = 12                # 每个桶最多带几条示例弹幕（过滤表情/去重后）
DANMAKU_MAX_TEXT = 40               # 示例弹幕最长字符数
DANMAKU_TOP_BUCKETS = 12            # 每块最多给大模型看几个最热桶
DANMAKU_PEAK_RATIO = 1.5            # 数量超过均值的多少倍算「高峰桶」

# ---------------------------------------------------------------------------
# 音频对齐参数（与 auto_edit.py 一致：剪辑点向外扩找静音）
# ---------------------------------------------------------------------------
SNAP_BOUNDARIES = True              # 开关：剪辑点是否向外扩到静音处
SNAP_MAX_SHIFT = 5.0                # 每个边界最多向外扩的秒数
MERGE_GAP_SEC = 1.0                 # 两段间隔小于这个秒数时合并（避免来回跳）
OVERFLOW_TOLERANCE_SEC = 30.0       # 音频对齐后允许超出成片上限的秒数（优先保完整）

# ---------------------------------------------------------------------------
# 剪辑倾向：向「手动剪辑」样本靠拢
# ---------------------------------------------------------------------------
# compare_edited_video.py 实测（手动剪辑 6 片 / 自动剪辑 4 片）：
#   内容占比  —— 手动：剧情/主线 42%、杂谈闲聊 33%、吐槽/玩梗 14%
#                自动：剧情/主线 85%、杂谈闲聊 1%（只挑剧情和战斗的结果）
#   主播占比  —— 手动：主播说话占语音 94%；自动：55%（选了很多纯游戏演出）
# 下面这组常量就是要补这两个差距：给候选打类型 / 主播占比，按配比重新配平。
#
# 只分 4 类：剧情主线 / 闲聊杂谈 / 吐槽玩梗 这三类在手动剪辑里样本足够，
# 其余（战斗、抽卡、读弹幕、攻略、纯情绪爆发……）单独统计样本太少、
# 大模型之间的边界也不稳定，统一并成「其他」，避免分类不准带偏配比。
CONTENT_TYPES = ["剧情/主线", "杂谈闲聊", "吐槽/玩梗", "其他"]

# 旧版 9 类 -> 新版 4 类（兼容老计划 / 大模型偶尔照抄旧类型名）
LEGACY_TYPE_MAP = {
    "剧情/主线": "剧情/主线",
    "杂谈闲聊": "杂谈闲聊",
    "吐槽/玩梗": "吐槽/玩梗",
    "战斗/操作": "其他",
    "抽卡/养成": "其他",
    "读弹幕互动": "其他",
    "攻略/讲解": "其他",
    "情绪高光": "其他",
    "其他": "其他",
}

# 类型定义：前 3 类要和 compare_edited_video.py 里的 TOPIC_GUIDE 保持一致，
# 否则两边对「杂谈 / 吐槽」的边界不一样，测出来的配比就对不上优化目标
CONTENT_TYPE_GUIDE = """- 剧情/主线：游戏剧情推进、主线对话、关键演出（主播在旁边解说/反应）
- 杂谈闲聊：主播随口聊日常、感想、和游戏无关的话题（没有明确的吐槽对象）
- 吐槽/玩梗：针对具体对象（剧情桥段、角色、弹幕、翻车事故）的吐槽、玩梗、
  整活、情绪化评价，主播在评论「某个东西」
- 其他：以上三类之外的一切——战斗/操作、抽卡/养成、读弹幕互动、攻略讲解、
  纯情绪爆发、跑图打怪。这类样本少不用细分，只挑确实精彩的"""

CONTENT_MIX_ENABLED = True          # 是否按目标内容配比校正
CONTENT_MIX = {                     # 先验配比（按时间占比），取自手动剪辑样本
    "剧情/主线": 0.40,
    "杂谈闲聊": 0.32,
    "吐槽/玩梗": 0.14,
    "其他": 0.14,
}
# 每类的动态区间（占成片时长的比例）：实际配比在候选池供给的基础上、
# 夹在这个区间里浮动，而不是死盯一个固定比例（某类好货不够就自然少给）
CONTENT_MIX_RANGE = {
    "剧情/主线": (0.20, 0.60),
    "杂谈闲聊": (0.15, 0.45),
    "吐槽/玩梗": (0.08, 0.30),
    "其他": (0.05, 0.35),
}
MIX_TOLERANCE = 0.05                # 每类允许偏离目标的比例（占目标总时长）
MIX_MAX_ITER = 50                  # 配平循环的最大迭代次数（每次只动一段）

STREAMER_TARGET_RATIO = 0.75        # 成片里主播说话占语音的目标比例（0 = 不约束）
STREAMER_TOLERANCE = 0.1            # 目标占比允许的偏差
MIN_CLIP_STREAMER_RATIO = 0.40      # 候选里主播说话占比低于这个就不选（0 = 不过滤）
TRIM_TO_STREAMER = False            # 片段首尾的大段非主播语音（纯游戏演出）裁掉
TRIM_MIN_SEC = 8.0                  # 首尾超过这个秒数的非主播语音才裁
TRIM_MAX_SEC = 90.0                 # 单侧最多裁多少秒

# ---------------------------------------------------------------------------
# 逻辑连接：把「主播吐槽的出处」并进片段（只在关联强的时候补，否则保持精简）
#   典型场景：主播说「这什么鬼提示啊」——上文 8 秒前刚出现的那句任务提示就是
#   被吐槽的对象，一起带上才看得懂；如果上文是在聊别的事，就不要往回拉。
# ---------------------------------------------------------------------------
LOGIC_LINK_ENABLED = True           # 开关：是否做上文出处拼接
LOGIC_BLOCK_GAP_SEC = 2.5           # 字幕间隔小于这个秒数算同一句「说话块」
LOGIC_LINK_LOOKBACK_SEC = 90.0      # 最多往前看多少秒
LOGIC_LINK_MAX_GAP_SEC = 25.0       # 上文和本段之间超过这个空档就不算「很近」
LOGIC_LINK_MAX_EXT_SEC = 60.0       # 单段最多往前补多少秒
LOGIC_LINK_MIN_OVERLAP = 0.10       # 开头用词和上文的二词重合度达到这个才算强关联
LOGIC_LINK_REF_GAP_SEC = 8.0        # 开头是「这个/刚才/你看」时，上文要在这个秒数内
LOGIC_LINK_REF_MIN_OVERLAP = 0.04   # 指代场景下重合度的下限（放宽一些）
LOGIC_LINK_REF_EXT_SEC = 25.0       # 指代场景下单次最多往前补多少秒
LOGIC_LINK_PAD_SEC = 0.4            # 补上文的起点再多留一点余量
LOGIC_LINK_EXTRA_BUDGET_SEC = 90.0  # 全片因补上文最多多花多少秒
LOGIC_OPEN_CHARS = 80               # 判定「本段开头在说什么」取多少字
LOGIC_OPEN_MAX_SEC = 30.0           # 上面那段开头文字最多往后取多少秒

# 运行时开关（main 根据命令行 / 字幕里有没有（主播）标记来覆盖）
PROFILE = {
    "has_streamer": False,         # 字幕里有没有主播标记（没有就跳过所有主播相关逻辑）
    "mix": CONTENT_MIX_ENABLED,
    "streamer_target": STREAMER_TARGET_RATIO,
    "min_clip_streamer": MIN_CLIP_STREAMER_RATIO,
    "trim_streamer": TRIM_TO_STREAMER,
}

# ---------------------------------------------------------------------------
# 成片重编码（修复 Premiere 时间轴错位）
#   stream copy 拼出来的 mp4 帧率不均匀（VFR）、时间戳是拼接式的，
#   Premiere 按平均帧率推算时间就会越往后越错位；整体重编码成 CFR 即可。
# ---------------------------------------------------------------------------
REENCODE_OUTPUT = True              # 开关：拼接完成后再整体重编码
REENCODE_CRF = 16                   # 画质档位（18 视觉上接近源片，越小越清晰）
REENCODE_PRESET = "slow"            # x264 速度档（veryfast 更快，slow 更省体积）
REENCODE_MAXRATE_RATIO = 2.0       # 码率上限 = 源视频码率 × 该系数（限制体积不膨胀）
REENCODE_AUDIO_BITRATE = 0          # 音频目标码率（bit/s），0 = 沿用源音频码率

# ---------------------------------------------------------------------------
# 成片字幕
# ---------------------------------------------------------------------------
WRITE_OUTPUT_SRT = True             # 开关：渲染成片时额外写一份对齐到成片的 SRT
MIN_OUT_SUB_SEC = 0.30              # 成片字幕最短显示时长，太短的（切点戈到的半句）丢掉

# ---------------------------------------------------------------------------
# 大模型参数（默认 DeepSeek，OpenAI 兼容接口）
# ---------------------------------------------------------------------------
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 180
LLM_CHUNK_SEC = 20 * 60             # 每次交给大模型的字幕时长（秒）
LLM_MAX_CANDIDATES_PER_CHUNK = 16   # 每块最多保留几个候选
LLM_FINAL_MAX_CANDIDATES = 140      # 全局挑选最多喂给大模型多少个候选
LLM_MAX_TOKENS = 8192
LLM_CONTEXT_SEC = 30.0              # 每块前后各附多少秒字幕作为上下文


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def fmt_sec(t):
    return f"{float(t):.1f}"


# ---------------------------------------------------------------------------
# SRT 解析
# ---------------------------------------------------------------------------
_SRT_TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[（(\[]\s*([^）)\]]{1,8})\s*[）)\]]\s*")


def _to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def read_srt(path):
    """返回 [{"start","end","text","speaker"}...]，按时间排序。"""
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw = f.read()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    entries = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        time_idx, match = None, None
        for i, line in enumerate(lines):
            m = _SRT_TIME_RE.search(line)
            if m:
                time_idx, match = i, m
                break
        if match is None:
            continue

        g = match.groups()
        start = _to_seconds(*g[:4])
        end = _to_seconds(*g[4:])
        text = " ".join(lines[time_idx + 1:]).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or end <= start:
            continue

        speaker = ""
        m = _SPEAKER_PREFIX_RE.match(text)
        if m:
            speaker = m.group(1).strip()
            text = text[m.end():].strip()
        if not text:
            continue

        entries.append({"start": start, "end": end,
                        "text": text, "speaker": speaker})

    entries.sort(key=lambda e: (e["start"], e["end"]))
    return entries


def clip_entries(entries, start, end):
    """裁剪到 [start, end]。"""
    out = []
    for e in entries:
        s = max(e["start"], start)
        t = min(e["end"], end)
        if t - s <= 0.05:
            continue
        item = dict(e)
        item["start"], item["end"] = s, t
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# 弹幕解析 / 对齐 / 聚合
# ---------------------------------------------------------------------------
def read_danmaku(path):
    """读取 B 站 xml，返回 [(真实时间秒, 文本)]。"""
    items = []
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "d":
            p = elem.get("p")
            text = (elem.text or "").strip()
            if p:
                try:
                    items.append((float(p.split(",", 1)[0]), text))
                except ValueError:
                    pass
            elem.clear()
    items.sort(key=lambda x: x[0])
    return items


def real_to_video_time(t, interval=DUP_INTERVAL, lag=DUP_LEN):
    """真实时间 -> 视频时间：每过一个重复点，视频就晚 lag 秒。"""
    if interval <= 0 or lag <= 0:
        return t
    return t + lag * int(t // interval)


_DANMAKU_EMOTE_RE = re.compile(r"\[[^\]]{1,40}\]")
_DANMAKU_WORD_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def clean_danmaku_text(text):
    """去掉弹幕表情（[xxx_xxx]）和多余空白，只留真正的文字。"""
    t = _DANMAKU_EMOTE_RE.sub("", text or "")
    return re.sub(r"\s+", " ", t).strip()


def build_danmaku_buckets(items, bucket_sec=DANMAKU_BUCKET_SEC):
    """把 [(视频时间秒, 文本)] 聚合成 {桶序号: {"start","count","text_count","samples"}}。

    计数包含所有弹幕（表情也是观众反应），但示例只保留文字弹幕并去重，
    这样同样的 prompt 预算能装进更多有效信息。
    """
    buckets = {}
    for t, text in items:
        idx = int(t // bucket_sec)
        item = buckets.setdefault(idx, {
            "start": idx * bucket_sec, "count": 0,
            "text_count": 0, "samples": []})
        item["count"] += 1

        cleaned = clean_danmaku_text(text)[:DANMAKU_MAX_TEXT]
        if not cleaned or not _DANMAKU_WORD_RE.search(cleaned):
            continue  # 纯表情 / 纯标点：只计入密度，不作为示例
        item["text_count"] += 1
        if cleaned in item["samples"] or len(item["samples"]) >= DANMAKU_SAMPLES:
            continue
        item["samples"].append(cleaned)
    return buckets


def format_danmaku(buckets, lo, hi, top_n=DANMAKU_TOP_BUCKETS):
    """给大模型看的弹幕热度摘要（带均值对比，突出异常高峰）。"""
    rows = []
    for data in buckets.values():
        start = data["start"]
        if start + DANMAKU_BUCKET_SEC < lo or start > hi:
            continue
        rows.append(data)
    if not rows:
        return "（本段没有弹幕数据）"

    counts = [d["count"] for d in rows]
    total = sum(counts)
    text_total = sum(d["text_count"] for d in rows)
    mean = total / len(counts)
    peak = max(counts)
    header = (
        f"本段 {lo:.0f}-{hi:.0f}s 共 {total} 条弹幕（其中文字 {text_total} 条，"
        f"平均每 {DANMAKU_BUCKET_SEC:g}s {mean:.1f} 条，峰值 {peak} 条）；"
        f"标注「高峰」的桶远高于本段均值，短弹幕（？/草/笑死）扎堆也常见于有节目效果的场面")

    rows.sort(key=lambda d: (-d["count"], d["start"]))
    lines = [header]
    for data in rows[:top_n]:
        start = data["start"]
        tag = ""
        if mean > 0 and data["count"] >= mean * DANMAKU_PEAK_RATIO:
            tag = f" 【高峰 ×{data['count'] / mean:.1f}】"
        sample = " / ".join(data["samples"]) if data["samples"] else "（无文字）"
        lines.append(f"{start:.0f}-{start + DANMAKU_BUCKET_SEC:.0f}s  "
                     f"{data['count']} 条{tag}: {sample}")
    if len(rows) > top_n:
        lines.append(f"（另有 {len(rows) - top_n} 个较冷清的桶未列出）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 多分片时间轴（一个文件夹里的多个视频 = 同一个视频被切成的分段）
# ---------------------------------------------------------------------------
def natural_key(name):
    """按「字母顺序 + 数字大小」排序：part2 排在 part10 前面。"""
    key = []
    for piece in re.split(r"(\d+)", str(name)):
        if not piece:
            continue
        if piece.isdigit():
            key.append((0, int(piece), ""))
        else:
            key.append((1, 0, piece.lower()))
    return key


def list_files(folder, exts):
    """列出文件夹里指定后缀的文件，按字母（数字）顺序排列。"""
    if not folder or not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder)
             if os.path.splitext(n)[1].lower() in exts
             and os.path.isfile(os.path.join(folder, n))]
    names.sort(key=natural_key)
    return [os.path.join(folder, n) for n in names]


def build_parts(video_paths):
    """把多个分片拼成一条全局时间轴。

    返回 [{"index", "path", "name", "duration", "offset"}]，
    offset 是这一段在全局时间轴上的起点（前面所有分片时长之和）。
    只有一个视频时同样适用，offset 恒为 0。
    """
    parts, offset = [], 0.0
    for i, path in enumerate(video_paths):
        dur = probe_duration(path)
        parts.append({
            "index": i,
            "path": path,
            "name": os.path.basename(path),
            "duration": dur,
            "offset": offset,
        })
        offset += dur
    return parts


def timeline_duration(parts):
    return sum(p["duration"] for p in parts)


def _part_index_at(parts, t):
    """全局时间 t 落在第几个分片里。"""
    for i, p in enumerate(parts):
        if t < p["offset"] + p["duration"] - 1e-9:
            return i
    return len(parts) - 1


def map_segment_to_parts(parts, start, end):
    """把全局区间拆成 [(part, 局部起点, 局部终点)]；跨分片时自动拆成两段。"""
    pieces = []
    for p in parts:
        s = max(float(start), p["offset"])
        e = min(float(end), p["offset"] + p["duration"])
        if e - s <= 0.02:
            continue
        pieces.append((p, s - p["offset"], e - p["offset"]))
    return pieces


def _norm_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", stem).lower()


def match_aux_files(parts, files, label):
    """把字幕 / 弹幕文件配到分片上，返回 {分片序号: 文件路径}。

    数量相同就按字母顺序一一对应（分片本身也是按字母顺序排的）；
    数量不同时先按文件名相似度配对，剩下的按顺序补齐。
    """
    if not files:
        return {}
    if len(files) == len(parts):
        return {i: files[i] for i in range(len(parts))}

    mapping, used = {}, set()
    for i, p in enumerate(parts):
        key = _norm_stem(p["path"])
        if not key:
            continue
        best, best_len = None, -1
        for j, f in enumerate(files):
            if j in used:
                continue
            fkey = _norm_stem(f)
            if not fkey:
                continue
            if key == fkey or key.startswith(fkey) or fkey.startswith(key):
                score = min(len(key), len(fkey))
                if score > best_len:
                    best, best_len = j, score
        if best is not None:
            used.add(best)
            mapping[i] = files[best]

    # 还有没配上的分片：按剩余文件的字母顺序补齐
    rest = [files[j] for j in range(len(files)) if j not in used]
    for i in range(len(parts)):
        if i in mapping or not rest:
            continue
        mapping[i] = rest.pop(0)

    if len(mapping) < len(parts):
        log(f"  ! {label}数量（{len(files)}）与分段数（{len(parts)}）不一致，"
            f"只配上了 {len(mapping)} 个分片")
    return mapping


def resolve_aux_source(source, exts):
    """解析 --srt / --danmaku：返回 (模式, 路径或文件列表)。

    "global"   = 单个文件，已经覆盖全部视频的时间轴；
    "per_part" = 文件夹，每个分片各一份；
    "none"     = 没有。
    """
    if not source:
        return "none", None
    if os.path.isdir(source):
        files = list_files(source, exts)
        return ("per_part", files) if files else ("none", None)
    if os.path.isfile(source):
        return "global", source
    return "none", None


def load_entries(parts, mode, source, v_start, v_end):
    """读出全局时间轴上的字幕（分片字幕会加上该分片的偏移）。"""
    if mode == "none":
        return []
    if mode == "global":
        return clip_entries(read_srt(source), v_start, v_end)

    files = match_aux_files(parts, source, "字幕")
    out = []
    for i, part in enumerate(parts):
        path = files.get(i)
        if not path:
            continue
        for e in read_srt(path):
            item = dict(e)
            item["start"] = e["start"] + part["offset"]
            item["end"] = e["end"] + part["offset"]
            item["part"] = i
            out.append(item)
    out.sort(key=lambda e: (e["start"], e["end"]))
    return clip_entries(out, v_start, v_end)


def load_danmaku(parts, mode, source):
    """读出弹幕并换算到全局视频时间，返回 [(视频时间秒, 文本)]。"""
    if mode == "none":
        return []
    if mode == "global":
        return [(real_to_video_time(t), text) for t, text in read_danmaku(source)]

    files = match_aux_files(parts, source, "弹幕")
    items = []
    for i, part in enumerate(parts):
        path = files.get(i)
        if not path:
            continue
        for t, text in read_danmaku(path):
            # 分P 的弹幕时间相对本分片；重复补偿的作用已经在分片偏移里
            items.append((part["offset"] + real_to_video_time(t), text))
    items.sort(key=lambda x: x[0])
    return items


# ---------------------------------------------------------------------------
# 大模型调用
# ---------------------------------------------------------------------------
def llm_chat(system_prompt, user_prompt):
    if not LLM_API_KEY:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY。请先设置环境变量，例如 PowerShell:\n"
            '    $env:DEEPSEEK_API_KEY="sk-xxxx"')

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        })
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"大模型接口 HTTP {exc.code}: {body[:500]}")
    except Exception as exc:
        raise RuntimeError(f"大模型调用失败: {exc}")

    return obj["choices"][0]["message"]["content"]


def parse_json_object(text):
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


# ---------------------------------------------------------------------------
# 分块 + 块内分析
# ---------------------------------------------------------------------------
def build_chunks(entries, chunk_sec):
    """按时间把字幕切成若干块。"""
    chunks = {}
    for e in entries:
        idx = int(e["start"] // chunk_sec)
        chunks.setdefault(idx, []).append(e)
    return [(idx * chunk_sec, idx * chunk_sec + chunk_sec, chunks[idx])
            for idx in sorted(chunks)]


def _nearest(values, t):
    if not values:
        return t
    i = bisect.bisect_left(values, t)
    if i <= 0:
        return values[0]
    if i >= len(values):
        return values[-1]
    return values[i] if values[i] - t < t - values[i - 1] else values[i - 1]


def _snap(values, t, max_delta=5.0):
    """把时间吸附到最近的字幕边界；离得太远就不动（防止片段被莫名拉长）。"""
    if not values:
        return t
    bound = _nearest(values, t)
    return bound if abs(bound - t) <= max_delta else t


def _speaker_entry(e):
    """这条字幕是不是主播自己说的（sensevoice_srt.py 打的（主播）标记）。"""
    return "主播" in (e.get("speaker") or "")


def _clean_type(value):
    """把大模型给的类型收敛到 4 类；旧类型名 / 简写也能认出来。"""
    value = str(value or "").strip()
    if value in CONTENT_TYPES:
        return value
    if value in LEGACY_TYPE_MAP:
        return LEGACY_TYPE_MAP[value]
    for key, t in (("剧情", "剧情/主线"), ("主线", "剧情/主线"),
                   ("杂谈", "杂谈闲聊"), ("闲聊", "杂谈闲聊"),
                   ("吐槽", "吐槽/玩梗"), ("玩梗", "吐槽/玩梗")):
        if key in value:
            return t
    return "其他"


def _mix_range_text(mix=None, ranges=None):
    """带上下限的配比说明（给大模型看）：剧情/主线 40%（20%~60%）。"""
    mix = mix if mix is not None else CONTENT_MIX
    ranges = ranges if ranges is not None else CONTENT_MIX_RANGE
    rows = []
    for t, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges.get(t, (0.0, 1.0))
        rows.append(f"{t} {w * 100:.0f}%（可浮动 {lo * 100:.0f}%~{hi * 100:.0f}%）")
    return "、".join(rows)


def _mix_quotas(total, mix=None):
    """把配比换算成「给几个候选」：按权重分配，总数不超过 total。"""
    mix = mix if mix is not None else CONTENT_MIX
    quotas, remain = {}, total
    for t, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        q = min(int(round(w * total)), remain)
        quotas[t] = max(q, 0)
        remain -= quotas[t]
    return quotas


def _clip_allocate(weights, lo, hi, total=1.0, iters=50):
    """把权重归一化后夹进 [lo, hi]，差额按「还有多少空间」补 / 扣回去。

    返回值不强求和为 total：如果所有类型都被 hi（候选池供给）卡住，
    剩下的就是「这块片子里根本凑不出来的量」，调用方据此知道配额取不满。
    """
    keys = list(weights)
    raw = {t: max(float(weights[t]), 0.0) for t in keys}
    s = sum(raw.values())
    w = ({t: raw[t] / s * total for t in keys} if s > 0
         else {t: total / len(keys) for t in keys})
    w = {t: min(max(w[t], lo[t]), hi[t]) for t in keys}

    for _ in range(iters):
        diff = total - sum(w.values())
        if abs(diff) < 1e-6:
            break
        if diff > 0:                      # 还差：按剩余空间 (hi - w) 加
            room = {t: hi[t] - w[t] for t in keys if hi[t] - w[t] > 1e-9}
            base = sum(room.values())
            if base <= 1e-9:
                break                     # 每类都到顶了：供给不够，配额只能少给
            step = min(diff, base)
            for t, r in room.items():
                w[t] += step * r / base
        else:                             # 超了：按可下降空间 (w - lo) 扣
            room = {t: w[t] - lo[t] for t in keys if w[t] - lo[t] > 1e-9}
            base = sum(room.values())
            if base <= 1e-9:
                break
            step = min(-diff, base)
            for t, r in room.items():
                w[t] -= step * r / base
    return w


def dynamic_mix(pool=None, budget=None, mix=None, ranges=None, floor_sec=None):
    """按候选池的实际供给动态分配 4 类内容的时长比例。

    先用手动剪辑样本的配比当先验，再按「这一片里每类实际有多少好货」调整，
    每类都夹在 CONTENT_MIX_RANGE 的区间里：
        剧情/主线 20%~60%、杂谈闲聊 15%~45%、吐槽/玩梗 8%~30%、其他 5%~35%
    某类候选严重不够时，下限让位给实际供给（否则配额永远取不满、
    只能拿不相关的片段硬凑）。传 None 池子时直接返回先验配比。

    floor_sec = 成片最短时长：上限卡得太死、按配额算出来连最短时长都凑不够时，
    以时长为先放宽上限（宁可配比差一点，也不要交一条过短的成片）。
    """
    mix = dict(mix or CONTENT_MIX)
    ranges = ranges if ranges is not None else CONTENT_MIX_RANGE
    types = list(CONTENT_TYPES)
    weights = {t: float(mix.get(t, 0.0)) for t in types}
    lo, hi = {}, {}
    for t in types:
        r = ranges.get(t, (0.0, 1.0))
        lo[t], hi[t] = float(r[0]), float(r[1])

    avail = {}
    if pool and budget and budget > 0:
        avail = {t: 0.0 for t in types}
        for c in pool:
            t = c.get("type") or "其他"
            avail[t] = avail.get(t, 0.0) + _dur(c)
        for t in types:
            cap = avail.get(t, 0.0) / float(budget)
            if cap < lo[t]:
                lo[t] = cap              # 供不上：下限让位给实际供给
            hi[t] = max(min(hi[t], cap), lo[t])
            if avail.get(t, 0.0) <= 0:   # 完全没货的类型不占配额
                weights[t] = 0.0

    shares = _clip_allocate(weights, lo, hi, 1.0)
    if (avail and budget and floor_sec and floor_sec > 0
            and sum(shares.values()) * budget < floor_sec
            and sum(avail.values()) >= floor_sec):
        shares = _clip_allocate(weights, lo, {t: 1.0 for t in types}, 1.0)
    return shares


def _quota_text(quotas):
    return "、".join(f"{t} {n} 个" for t, n in quotas.items() if n > 0)


def clip_streamer_stats(entries, start, end):
    """统计 [start, end] 里主播说话时长 / 语音总时长 / 字数。"""
    speech = streamer = 0.0
    chars = 0
    for e in entries:
        s, t = max(e["start"], start), min(e["end"], end)
        if t - s <= 0:
            continue
        speech += t - s
        if _speaker_entry(e):
            streamer += t - s
        chars += len(e.get("text") or "")
    return {"speech_sec": speech, "streamer_sec": streamer, "chars": chars,
            "streamer_ratio": (streamer / speech) if speech > 0 else 0.0}


def trim_to_streamer(entries, start, end, min_clip):
    """把片段首尾的大段非主播语音（纯游戏演出）裁掉，只留主播说话的部分。

    只裁超过 TRIM_MIN_SEC 的边缘，单侧最多裁 TRIM_MAX_SEC，裁完还得够 min_clip，
    所以不会把「主播在旁边听剧情」这种正常片段弄碎。
    """
    if not PROFILE["trim_streamer"] or not PROFILE["has_streamer"]:
        return start, end
    spans = [e for e in entries if e["end"] > start and e["start"] < end]
    streamer = [e for e in spans if _speaker_entry(e)]
    if not streamer:
        return start, end

    pad = 0.4
    ns = max(start, streamer[0]["start"] - pad)
    ne = min(end, streamer[-1]["end"] + pad)
    if ns - start < TRIM_MIN_SEC and end - ne < TRIM_MIN_SEC:
        return start, end                   # 首尾都只是零头，不用裁
    if ns - start > TRIM_MAX_SEC:
        ns = start + TRIM_MAX_SEC
    if end - ne > TRIM_MAX_SEC:
        ne = end - TRIM_MAX_SEC
    if ne - ns < min_clip:
        return start, end
    return ns, ne


def annotate_candidates(entries, items, min_clip):
    """给候选补上「主播说话占比 / 字数 / 时长」，并把首尾的游戏演出裁掉。"""
    out = []
    for c in items:
        s, e = trim_to_streamer(entries, c["start"], c["end"], min_clip)
        item = dict(c)
        item["start"], item["end"] = s, e
        item.update(clip_streamer_stats(entries, s, e))
        item["dur"] = e - s
        out.append(item)
    return out


def filter_by_streamer(items):
    """主播说话占比太低的候选直接丢掉（但每块至少留 2 个，别把候选池抽干）。"""
    floor = PROFILE["min_clip_streamer"]
    if not PROFILE["has_streamer"] or floor <= 0 or not items:
        return items
    kept = [c for c in items if c["streamer_ratio"] >= floor]
    if len(kept) >= 2:
        return kept
    kept_ids = {id(c) for c in kept}
    rest = sorted((c for c in items if id(c) not in kept_ids),
                  key=lambda c: -(c["streamer_ratio"] * 10 + c.get("score", 0)))
    return kept + rest[:max(0, 2 - len(kept))]


def normalize_candidates(items, entries, lo, hi, min_clip):
    """把大模型给的候选对齐到字幕边界，并做基本校验。"""
    starts = {e["start"] for e in entries}
    boundaries = sorted(starts | {e["end"] for e in entries})
    out = []
    for item in items or []:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue

        start = _snap(boundaries, start) if starts else start
        end = _snap(boundaries, end) if boundaries else end
        start = max(lo, min(start, hi))
        end = max(lo, min(end, hi))

        if end - start < min_clip:
            end = min(hi, start + min_clip)
        if end - start < min_clip:
            continue

        try:
            score = float(item.get("score", 5))
        except (TypeError, ValueError):
            score = 5.0

        out.append({
            "start": start,
            "end": end,
            "score": max(0.0, min(score, 10.0)),
            "type": _clean_type(item.get("type")),
            "title": str(item.get("title", "")).strip()[:40],
            "reason": str(item.get("reason", "")).strip()[:120],
            "text": str(item.get("corrected_text", "")).strip()[:120],
        })
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


CHUNK_SYSTEM = """你是游戏直播回放的精剪助手。你会看到一段带时间戳的字幕
（本段前后各附少量标注为「上下文」的行），以及可选的弹幕热度摘要。
字幕来自语音识别（已在字幕阶段做过一轮纠错），仍可能有错。
行首标了「[主播]」的是主播自己说的，没标的是游戏 NPC / 剧情演出 / 其他声音
（声纹判断不一定 100% 准，但整段都没有 [主播] 的，基本就是纯游戏声音）。
弹幕在问答时也会很密集，不能只看密度，但某个时间桶的弹幕量明显高于本段均值时，
通常说明那一刻有节目效果（哪怕弹幕内容只是「？」「草」「哈哈哈」）。

请完成两件事（一步完成）：
1. 结合上下文顺手修正仍明显出错的台词（只写进 corrected_text，不要求逐条）；
2. 挑出值得剪进精华片的候选片段。

硬性要求：
- 只输出 JSON：{{"items": [{{"start": 秒, "end": 秒, "title": "短标题",
  "type": "内容类型", "reason": "为什么值得剪", "score": 0-10, "corrected_text": "纠正后的关键台词"}}]}}
- type 只能从这些里选：{content_types}；按下面的定义判断，不要都写成剧情/主线：
{content_guide}
- start / end 必须直接取自本段字幕里出现过的起止时间，不要自己编造，
  也不要越出本段范围（「上下文」行不能作为边界）；
- 片段要尽量语义完整：起点落在话题/句子开头，终点落在说完之后，
  不要从半句话中间开始或结束；宁可多带一点前因后果；
- **逻辑连接（很重要）**：主播在吐槽 / 评论某个信息时，如果那条信息就在上文
  很近的地方（同一话题的上一句、刚出现的提示/字幕/剧情），就把那条信息
  一起包进片段，起点提到信息出现之前，让观众看得懂他在骂什么；
  反之，如果找上文的关联内容离得远（-> {logic_gap}s）、或者和本段话题对不上，
  就**不要**往回带，直接从主播这段话自己的起点开始（宁可短一点，不要夹无关内容）；
- **精简**：片段里不要夹和主题无关的空转（纯跑图、无解说演出、重复操作），
  但也不能为了缩短把一句话 / 一个包袱切一半；
- 每个片段 end - start >= {min_clip} 秒，且必须落在本段范围内；
- 按时间顺序输出，最多 {max_items} 个，score 越高越优先；
- **优先选主播本人在说话、有解说、有反应的片段**：整段几乎全是游戏过场语音
  （没有 [主播] 标记）的片段不要选；边界尽量落在主播的句子之间；
- 成片的目标内容配比（按时间占比）：{content_mix}。
  「杂谈闲聊 / 吐槽玩梗」在手动剪辑里占比很高，不要只挑剧情和战斗；
  剧情/主线只挑主播有解说、有反应的关键节点，整段过场动画不算剧情高光；
- 本段候选的类型个数尽量按这个配额给：{chunk_quota}
  （本段凑不出来的类型可以少给，但不要整段全是同一类，尤其是不要全给剧情）；
- 用户偏好：{edit_prompt}
"""


def analyze_chunk(entries, lo, hi, danmaku_text, min_clip, max_items, edit_prompt,
                  context_sec=LLM_CONTEXT_SEC):
    """分析一块字幕。entries 传全量字幕，本块之外的只作为上下文。"""
    inside = [e for e in entries if e["end"] > lo and e["start"] < hi]
    if not inside:
        return []

    def render(e, is_context):
        tag = "[主播]" if _speaker_entry(e) else ""
        prefix = "[上下文] " if is_context else ""
        return f"{prefix}[{e['start']:.1f}-{e['end']:.1f}]{tag} {e['text']}"

    before = [e for e in entries if lo - context_sec <= e["end"] <= lo]
    after = [e for e in entries if hi <= e["start"] <= hi + context_sec]
    subtitle_text = "\n".join(
        [render(e, True) for e in before[-8:]] +
        [render(e, False) for e in inside] +
        [render(e, True) for e in after[:8]])

    mix_hint = (_mix_range_text() if PROFILE["mix"]
                else "不限定（按精彩程度挑）")
    quota_hint = (_quota_text(_mix_quotas(max_items)) if PROFILE["mix"]
                  else f"不超过 {max_items} 个")
    system = CHUNK_SYSTEM.format(
        min_clip=f"{min_clip:g}",
        max_items=max_items,
        content_types="、".join(CONTENT_TYPES),
        content_guide=CONTENT_TYPE_GUIDE,
        content_mix=mix_hint,
        chunk_quota=quota_hint,
        logic_gap=f"{LOGIC_LINK_MAX_GAP_SEC:g}",
        edit_prompt=edit_prompt or "无",
    )
    user = (f"本段范围：{lo:.1f}s - {hi:.1f}s（片段 start/end 必须落在这段范围内）\n\n"
            f"字幕：\n{subtitle_text}\n\n"
            f"弹幕热度（仅作辅助）：\n{danmaku_text}")

    try:
        raw = llm_chat(system, user)
        obj = parse_json_object(raw)
    except Exception as exc:
        log(f"  ! 分段分析失败（{lo:.0f}-{hi:.0f}s）：{exc}")
        return []

    found = normalize_candidates(obj.get("items", []), inside, lo, hi, min_clip)
    found = annotate_candidates(entries, found, min_clip)
    raw_count = len(found)
    found = filter_by_streamer(found)
    if raw_count != len(found):
        log(f"    · 主播过滤：{raw_count} -> {len(found)} 个候选")
    found.sort(key=lambda x: -x["score"])
    found = found[:max_items]
    found.sort(key=lambda x: x["start"])
    return found


# ---------------------------------------------------------------------------
# 全局挑选 + 硬约束
# ---------------------------------------------------------------------------
FINAL_SYSTEM = """你是总剪辑师。输入是按时间排序的候选片段列表，
每条的「类型」是大模型判定的内容类型，「主播」是这段里主播说话时长占语音的比例。
每条还附了片段附近的字幕（行首「>」的在本片段内部，其余为前后上下文）。
请从中挑选一组片段，要求：
1. 按时间顺序、互不重叠；
2. 每个片段 duration >= {min_clip} 秒；
3. 总时长在 {target_min} - {target_max} 秒之间（尽量接近中间值）；
4. 内容配比尽量贴近（按时间占比）：{content_mix}；
5. 主播说话占比要高：全片平均不低于 {streamer_target}%，
   「主播」明显偏低的候选（纯游戏演出）尽量别选；
6. 优先选 score 高、信息量大、能体现主播本人表达和反应的片段；
7. 如果字幕上下文显示某候选从半句话开始、话没说完就结束，
   或与另一候选是同一段剧情的前后两截，避开它或只保留更完整的一个；
8. 避免全片都是同一主题，也不要整片都是纯剧情过场（没有主播解说）。
9. **逻辑连贯优先**：优先保留「上文给了信息 -> 主播吐槽 / 反应」这种成对的片段；
   相邻候选如果话题连得上、中间只隔几秒，当作一段连续内容一起选（不要切成两截）；
   找不到前因的孤立反应（上来就「这太好笑了」「什么玩意」）尽量别选，
   除非它本身就完整；
10. **精简但不破碎**：关联不强的前后文不要为了凑时长一起选，
    但也不要为了缩短时长挑半句话的片段。

只输出 JSON：{{"selected": [候选编号, ...]}}，按时间顺序排列。
"""


def _excerpt(entries, start, end, pad=20.0, max_lines=6):
    """取片段附近的字幕，行首 > 标记片段内部的行。"""
    rows = []
    for e in entries:
        if e["end"] < start - pad or e["start"] > end + pad:
            continue
        inside = e["start"] >= start - 0.05 and e["end"] <= end + 0.05
        rows.append((">" if inside else " ") + e["text"])
    if not rows:
        return ""
    if len(rows) > max_lines:
        head = max_lines // 2
        rows = rows[:head] + ["……"] + rows[-(max_lines - head):]
    return " / ".join(rows)


def select_with_llm(candidates, entries, target_min, target_max, min_clip):
    rows = []
    for c in candidates:
        rows.append(
            f"#{c['id']} {c['start']:.1f}-{c['end']:.1f} "
            f"({c['end'] - c['start']:.1f}s) score={c['score']:.0f} "
            f"类型={c.get('type') or '其他'} "
            f"主播={c.get('streamer_ratio', 0) * 100:.0f}% "
            f"| {c['title']} | {c['reason']}"
        )
        excerpt = _excerpt(entries, c["start"], c["end"])
        if excerpt:
            rows.append(f"    字幕: {excerpt}")
    mix_hint = (_mix_range_text() if PROFILE["mix"]
                else "不限定（按精彩程度挑）")
    system = FINAL_SYSTEM.format(min_clip=f"{min_clip:g}",
                                 target_min=f"{target_min:g}",
                                 target_max=f"{target_max:g}",
                                 content_mix=mix_hint,
                                 streamer_target=f"{PROFILE['streamer_target'] * 100:.0f}")
    user = ("候选片段：\n" + "\n".join(rows) +
            f"\n\n请选出总时长 {target_min:g}-{target_max:g} 秒的一组片段。")

    try:
        obj = parse_json_object(llm_chat(system, user))
        ids = [int(x) for x in obj.get("selected", [])]
    except Exception as exc:
        log(f"  ! 全局挑选失败，改用分数排序：{exc}")
        return []

    by_id = {c["id"]: c for c in candidates}
    return [by_id[i] for i in ids if i in by_id]


def fit_selection(candidates, chosen, target_min, target_max,
                  min_clip, v_start, v_end):
    """保证最终时间轴合法：顺序、不重叠、单段够长、总时长在范围内。"""
    def clamp(rec):
        s = max(v_start, float(rec["start"]))
        e = min(v_end, float(rec["end"]))
        if e - s < min_clip:
            e = min(v_end, s + min_clip)
        return s, e

    def resolve(items):
        items = sorted(items, key=lambda x: x["start"])
        out = []
        for it in items:
            if out and it["start"] < out[-1]["end"]:
                it = dict(it)
                it["start"] = out[-1]["end"]
            if it["end"] - it["start"] >= min_clip:
                out.append(it)
        return out

    picked, used = [], set()
    for rec in chosen or []:
        s, e = clamp(rec)
        if e - s < min_clip or rec["id"] in used:
            continue
        used.add(rec["id"])
        item = dict(rec)
        item["start"], item["end"] = s, e
        picked.append(item)

    picked = resolve(picked)
    total = sum(x["end"] - x["start"] for x in picked)

    # 不够长：按分数补
    if total < target_min:
        pool = sorted((c for c in candidates if c["id"] not in used),
                      key=lambda x: -x["score"])
        for rec in pool:
            if total >= target_min:
                break
            s, e = clamp(rec)
            if e - s < min_clip:
                continue
            if any(not (e <= p["start"] or s >= p["end"]) for p in picked):
                continue
            item = dict(rec)
            item["start"], item["end"] = s, e
            picked.append(item)
            used.add(rec["id"])
            total += e - s

    # 超长：先削最低分片段的尾巴，削不动就删
    picked.sort(key=lambda x: x["start"])
    total = sum(x["end"] - x["start"] for x in picked)
    while total > target_max and picked:
        idx = min(range(len(picked)), key=lambda i: picked[i]["score"])
        rec = picked[idx]
        excess = total - target_max
        if rec["end"] - rec["start"] - excess >= min_clip:
            rec["end"] -= excess
            total -= excess
        elif len(picked) == 1 and target_max >= min_clip:
            rec["end"] = rec["start"] + target_max
            total = target_max
            break
        else:
            total -= rec["end"] - rec["start"]
            picked.pop(idx)
        if len(picked) == 1 and total <= target_max:
            break

    picked.sort(key=lambda x: x["start"])
    return picked


# ---------------------------------------------------------------------------
# 剪辑倾向校正：内容配比 + 主播说话占比
# ---------------------------------------------------------------------------
def _dur(seg):
    return max(float(seg["end"]) - float(seg["start"]), 0.0)


def _pool_ratio(segs):
    """已选片段的整体主播说话占比（按语音时长加权）。"""
    speech = sum(_dur(s) for s in segs)
    if speech <= 0:
        return 0.0
    streamer = sum(float(s.get("streamer_sec") or 0.0) for s in segs)
    return min(streamer / speech, 1.0)


def shortlist_candidates(candidates, limit=LLM_FINAL_MAX_CANDIDATES):
    """全局挑选前按内容配额截断候选：保证每类都有好货，提示词也不会太长。

    配额用 dynamic_mix 按候选池实际供给动态算，而不是死守固定比例：
    某类根本没多少候选时，配额自然少给，把名额让给别的类型（否则结果里
    会充满低分的凑数片段）。
    """
    items = list(candidates)
    if len(items) <= limit or limit <= 0:
        return items
    rank = lambda c: -(float(c.get("score", 5)) + 4.0 * float(c.get("streamer_ratio") or 0))
    if not PROFILE["mix"]:
        return sorted(items, key=rank)[:limit]

    shares = dynamic_mix(items, sum(_dur(c) for c in items) or 1.0)
    by_type = {}
    for c in sorted(items, key=rank):
        by_type.setdefault(c.get("type") or "其他", []).append(c)
    picked = []
    for t, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        want = int(round(share * limit))
        if want <= 0:
            continue
        picked.extend((by_type.get(t) or [])[:want])
    if len(picked) < limit:                 # 某类候选不够，用剩下的高分补齐
        picked_ids = {c.get("id") for c in picked}
        picked.extend([c for c in sorted(items, key=rank)
                       if c.get("id") not in picked_ids][:limit - len(picked)])
    picked.sort(key=lambda c: (c["start"], c["end"]))
    return picked


def align_to_profile(segments, candidates, target_min, target_max, min_clip):
    """按「内容配比 + 主播占比」重建挑选结果（向手动剪辑样本靠拢）。

    做法是「配额重建」而不是反复增删，避免来回震荡：
      1. 预算 T = 当前总时长（夹在 target_min ~ target_max 之间）；
      2. **动态配额**：按先验配比 + 候选池里每类的实际供给，算出每类该占多少秒
         （每类夹在 CONTENT_MIX_RANGE 区间里，比如主线 20%~60%、
         杂谈 15%~45%），从候选池里依次取（大模型选过的优先，
         其次分数 + 主播占比高的），同类型取到配额为止；
      3. 配额取完还有富余，用最优候选补到 T（候选不够就算了）；
      4. 超长时先砍超额类别（按动态配额判定超额，而不是固定比例）；
      5. 主播说话占比不达标时，用高主播候选把最低的几段换掉。
    每步都是确定性的，不会来回震荡；配比 / 主播目标都在文件头常量里。
    """
    if not segments:
        return []
    target_ratio = PROFILE["streamer_target"] if PROFILE["has_streamer"] else 0.0

    budget = min(max(sum(_dur(s) for s in segments), target_min), target_max)
    chosen_ids = {s.get("id") for s in segments if s.get("id") is not None}

    def value(c):
        return (1 if c.get("id") in chosen_ids else 0,
                float(c.get("score", 5)) + 4.0 * float(c.get("streamer_ratio") or 0))

    pool = [c for c in candidates if _dur(c) >= min_clip]
    pool.sort(key=value, reverse=True)

    # 4 类内容的比例不是写死的：先验配比 + 本片候选池的实际供给，夹到各自的上下限
    mix = (dynamic_mix(pool, budget, floor_sec=target_min)
           if PROFILE["mix"] else {})
    if mix:
        log(f"  · 动态配额：{_fmt_mix(mix)}")

    kept, used = [], set()

    def fits(c, skip=None):
        return not any(k is not skip and
                       not (c["end"] <= k["start"] or c["start"] >= k["end"])
                       for k in kept)

    def use(c):
        kept.append(dict(c))
        used.add(c.get("id"))

    def sec_of(t):
        return sum(_dur(k) for k in kept if (k.get("type") or "其他") == t)

    # ---- 1) 按配额取每类 ----
    for t, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        quota = budget * w
        if quota < min_clip:
            continue
        for c in pool:
            remaining = quota - sec_of(t)
            if remaining <= min_clip * 0.5:
                break
            if c.get("id") in used or (c.get("type") or "其他") != t or not fits(c):
                continue
            if _dur(c) > remaining + min_clip:
                continue
            use(c)

    # ---- 2) 补满到预算（已经超配额的类别尽量不再补，避免配比被带偏）----
    total = sum(_dur(k) for k in kept)

    def fill_to_budget(strict=True):
        nonlocal total
        for c in pool:
            if total >= budget:
                return
            if c.get("id") in used or not fits(c):
                continue
            t = c.get("type") or "其他"
            if strict and mix:
                quota = budget * (mix.get(t) or 0.0)
                # 配额几乎为 0 的类型（候选池本来就少）不再拿来凑数
                if quota < min_clip or sec_of(t) >= quota:
                    continue
            use(c)
            total += _dur(c)

    fill_to_budget(strict=True)
    if total < target_min:                 # 挑不出合适类型了：先保证成片够长
        fill_to_budget(strict=False)

    # ---- 3) 超长：先砍超额类别里分最低的 ----
    while total > target_max and len(kept) > 1:
        over_types = {t for t, w in mix.items()
                      if sec_of(t) > budget * w + budget * MIX_TOLERANCE}
        cands = [k for k in kept if (k.get("type") or "其他") in over_types] or kept
        victim = min(cands, key=lambda s: (float(s.get("score", 5)),
                                           float(s.get("streamer_ratio") or 0)))
        if total - _dur(victim) < target_min:
            break
        total -= _dur(victim)
        kept.remove(victim)
        used.discard(victim.get("id"))

    # ---- 4) 主播说话占比不够：换掉占比最低的几段 ----
    swaps = 0
    for _ in range(MIX_MAX_ITER):
        if target_ratio <= 0 or _pool_ratio(kept) >= target_ratio - STREAMER_TOLERANCE:
            break
        victim = min(kept, key=lambda s: (float(s.get("streamer_ratio") or 0),
                                          float(s.get("score", 5))))
        v_ratio = float(victim.get("streamer_ratio") or 0)
        best, fallback = None, None
        for c in pool:
            if c.get("id") in used:
                continue
            if float(c.get("streamer_ratio") or 0) <= v_ratio + 0.15:
                continue
            if not fits(c, skip=victim):
                continue
            # 优先换进「还没超配额」的类型，别让主播替换把 4 类配比带偏
            t = c.get("type") or "其他"
            if mix and sec_of(t) >= budget * (mix.get(t) or 0.0) + min_clip:
                if fallback is None:
                    fallback = c
                continue
            best = c
            break
        best = best or fallback
        if best is None:
            break
        kept.remove(victim)
        used.discard(victim.get("id"))
        use(best)
        swaps += 1

    if not kept:
        return [dict(s) for s in segments]

    kept.sort(key=lambda s: (s["start"], s["end"]))
    log(f"  · 倾向校正：{len(segments)} -> {len(kept)} 段"
        f"（按内容配比重建，主播片段替换 {swaps} 段）")
    return kept


def segment_profile(entries, segments):
    """按最终片段算一遍「主播说话占语音」和「内容类型占比」（给日志/计划用）。"""
    speech = streamer = 0.0
    mix = {}
    for seg in segments:
        st = clip_streamer_stats(entries, seg["start"], seg["end"])
        speech += st["speech_sec"]
        streamer += st["streamer_sec"]
        t = seg.get("type") or "其他"
        mix[t] = mix.get(t, 0.0) + _dur(seg)
    total = sum(mix.values()) or 1.0
    return {"streamer": (streamer / speech) if speech > 0 else 0.0,
            "mix": {t: v / total for t, v in
                    sorted(mix.items(), key=lambda kv: -kv[1])}}


def _fmt_mix(mix):
    return " · ".join(f"{t} {w * 100:.0f}%" for t, w in mix.items())


# ---------------------------------------------------------------------------
# 逻辑连接：把「主播吐槽 / 评论的出处」并进片段
#   「这什么鬼提示啊」这类反应，观众得先看到 8 秒前那条提示才看得懂，
#   所以只要上文和本段明显在说同一件事（关键词接续 / 指代上一句），
#   就把上文一起带上；关联不强的上下文坚决不补，保证时间紧凑。
# ---------------------------------------------------------------------------
_GRAM_CLEAN_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
# 这些两字组合到处都是，不能当成「在说同一件事」的证据
_STOP_GRAMS = {
    "这个", "那个", "就是", "然后", "所以", "因为", "但是", "我们", "你们",
    "他们", "现在", "时候", "真的", "可以", "感觉", "知道", "什么", "怎么",
    "一下", "已经", "还是", "不过", "其实", "应该", "觉得", "有点", "不是",
    "没有", "这么", "那么", "一个", "这种", "那种", "这样", "那样", "这里",
    "那里", "东西", "直接", "看到", "去看", "完了", "还有",
}
# 明确指向「上文刚出现的东西」的说法（不包含「他 / 这」这类过于常见的词）
_REF_WORD_RE = re.compile(
    r"这个|那个|刚才|刚刚|前面|上面|你看|这波|这玩意|这什么|这提示|这句|这俩|这一下")


def _text_grams(text):
    """把文字拆成两字词集合（去掉泛用词），用来判断两段话是不是同一件事。"""
    clean = _GRAM_CLEAN_RE.sub("", text or "")
    grams = {clean[i:i + 2] for i in range(len(clean) - 1)}
    return {g for g in grams if g not in _STOP_GRAMS}


def build_speech_blocks(entries, gap_sec=LOGIC_BLOCK_GAP_SEC):
    """把字幕按停顿合成「说话块」，作为补齐上文时的最小单位。"""
    blocks = []
    for e in entries:
        if blocks and e["start"] - blocks[-1]["end"] <= gap_sec:
            blocks[-1]["end"] = max(blocks[-1]["end"], e["end"])
            blocks[-1]["items"].append(e)
        else:
            blocks.append({"start": e["start"], "end": e["end"], "items": [e]})
    for blk in blocks:
        blk["text"] = "".join(i.get("text") or "" for i in blk["items"])
    return blocks


def _opening_text(entries, start, chars=LOGIC_OPEN_CHARS,
                  max_sec=LOGIC_OPEN_MAX_SEC):
    """取片段开头的一小段文字，代表「这段话在说什么」。"""
    out, total = [], 0
    for e in entries:
        if e["end"] <= start:
            continue
        if e["start"] - start > max_sec:
            break
        out.append(e.get("text") or "")
        total += len(e.get("text") or "")
        if total >= chars:
            break
    return "".join(out)[:chars]


def find_context_start(blocks, start, open_text, prev_end=None, max_ext=None):
    """找「本段开头提到的信息」在上文的出处。

    返回 (新起点, 说明) 或 None。判定两种：
      1. 关键词接续：开头用词和上文的二词重合度 >= LOGIC_LINK_MIN_OVERLAP；
      2. 指代上一句：开头是「这个/刚才/你看」之类，且上一句就在几秒内。
    离得远（> LOGIC_LINK_MAX_GAP_SEC）、或者已经被前面的片段用掉的，不补。
    """
    max_ext = LOGIC_LINK_MAX_EXT_SEC if max_ext is None else max_ext
    open_grams = _text_grams(open_text)
    if not open_grams:
        return None

    floor = start - LOGIC_LINK_LOOKBACK_SEC
    lo_bound = (prev_end + 0.05) if prev_end is not None else None
    run, cursor = [], start

    # 1) 本段所在说话块里、位于起点之前的部分（同一句话的前半截也要算上文）
    host = next((b for b in blocks
                 if b["start"] < start - 0.05 and b["end"] > start - 0.05), None)
    if host is not None:
        head_start = host["start"]
        if lo_bound is not None:
            head_start = max(head_start, lo_bound)
        head_text = "".join(i.get("text") or "" for i in host["items"]
                            if head_start - 0.05 <= i["start"] < start - 0.05)
        if head_text:
            run.append({"start": head_start, "end": start, "text": head_text})
            cursor = head_start

    # 2) 再往前整块整块地捡，断档太大 / 碰到前一段就停
    for blk in reversed(blocks):
        if blk["end"] > cursor + 0.05:
            continue                     # 已经把这块算进来了（或是本段自己）
        if blk["end"] < floor:
            break
        if lo_bound is not None and blk["start"] < lo_bound - 0.05:
            break                        # 这块已经在前面的片段里了
        if cursor - blk["end"] > LOGIC_LINK_MAX_GAP_SEC:
            break                        # 中间断档太大，不算「上文很近」
        run.append(blk)
        cursor = blk["start"]
        if len(run) >= 7 or start - blk["start"] > max_ext:
            break
    if not run:
        return None

    nearest_gap = start - run[0]["end"]
    for n in range(1, len(run) + 1):
        chunk = run[:n]                  # 离本段最近的 n 块
        grams = _text_grams("".join(b["text"] for b in chunk))
        if not grams:
            continue
        overlap = len(open_grams & grams) / len(open_grams)
        new_start = min(b["start"] for b in chunk) - LOGIC_LINK_PAD_SEC
        new_start = max(new_start, start - max_ext)
        if lo_bound is not None:
            new_start = max(new_start, lo_bound)
        if overlap >= LOGIC_LINK_MIN_OVERLAP and start - new_start >= 1.0:
            return new_start, f"关键词接续 {overlap * 100:.0f}%"
        if (n == 1 and nearest_gap <= LOGIC_LINK_REF_GAP_SEC
                and _REF_WORD_RE.search(open_text)
                and overlap >= LOGIC_LINK_REF_MIN_OVERLAP):
            new_start = max(new_start, start - LOGIC_LINK_REF_EXT_SEC)
            if start - new_start >= 1.0:
                return new_start, f"指代上文（{nearest_gap:.0f}s 前）"
    return None


def stitch_context(segments, entries, max_total=None):
    """逐段往前补上「被吐槽的出处」，补出来的时长有总预算，不会把成片撑长。"""
    if not LOGIC_LINK_ENABLED or not segments:
        return segments, {"extended": 0, "extra": 0.0}

    ordered = sorted(segments, key=lambda x: (x["start"], x["end"]))
    total = sum(_dur(s) for s in ordered)
    cap = total + LOGIC_LINK_EXTRA_BUDGET_SEC if max_total is None else max_total
    room = min(LOGIC_LINK_EXTRA_BUDGET_SEC, cap - total)
    blocks = build_speech_blocks(entries)

    out, extended, extra, prev_end = [], 0, 0.0, None
    for seg in ordered:
        item = dict(seg)
        start = float(item["start"])
        if room > 1.0:
            found = find_context_start(
                blocks, start, _opening_text(entries, start),
                prev_end=prev_end, max_ext=min(LOGIC_LINK_MAX_EXT_SEC, room))
            if found:
                new_start, why = found
                gain = start - new_start
                if 0.5 < gain <= room:
                    item["start"] = new_start
                    item["ctx_sec"] = round(gain, 1)
                    item["ctx_reason"] = why
                    extended += 1
                    extra += gain
                    room -= gain
        out.append(item)
        prev_end = float(item["end"])
    return out, {"extended": extended, "extra": extra}


# ---------------------------------------------------------------------------
# 音频对齐：剪辑点向外扩到静音处（参考 auto_edit.py）
# ---------------------------------------------------------------------------
def snap_segments(parts, segments, max_shift=SNAP_MAX_SHIFT,
                  v_start=0.0, v_end=None):
    """每个片段的开始点往前、结束点往后扩，切到附近最安静的位置。

    多分片时，开始 / 结束点各自在自己所在的分片里找静音，不会跨过
    分片边界；跨分片的片段在边界处本身就是合法切点。
    """
    out, moved = [], 0
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        sp = parts[_part_index_at(parts, s)]
        ep = parts[_part_index_at(parts, max(s, e - 1e-6))]
        ns = sp["offset"] + snap_cut(sp["path"], s - sp["offset"], -1,
                                     max_shift, sp["duration"])
        ne = ep["offset"] + snap_cut(ep["path"], e - ep["offset"], +1,
                                     max_shift, ep["duration"])
        ns = max(v_start, ns)
        if v_end is not None:
            ne = min(v_end, ne)
        if ne - ns < 0.05:
            ns, ne = s, e

        item = dict(seg)
        item["orig_start"], item["orig_end"] = s, e
        item["start"], item["end"] = ns, ne
        out.append(item)
        if abs(ns - s) > 1e-3 or abs(ne - e) > 1e-3:
            moved += 1
    return out, moved


def merge_segments(segments, gap=MERGE_GAP_SEC):
    """合并向外扩后相互重叠（或几乎相接）的片段，保留更完整的那条的说明。"""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: (x["start"], x["end"]))
    merged = [dict(ordered[0])]
    for it in ordered[1:]:
        last = merged[-1]
        if it["start"] > last["end"] + gap:
            merged.append(dict(it))
            continue
        last["end"] = max(last["end"], it["end"])
        last["orig_end"] = max(last.get("orig_end", last["end"]),
                               it.get("orig_end", it["end"]))
        if it.get("score", 0) > last.get("score", 0):
            for key in ("score", "title", "reason", "text"):
                if it.get(key):
                    last[key] = it[key]
    return merged


def enforce_duration(segments, target_min, target_max, min_clip):
    """外扩可能把总时长顶出上限；尽量只收回多扩的部分，保住整段。"""
    segs = [dict(s) for s in segments]
    total = sum(s["end"] - s["start"] for s in segs)
    if total <= target_max:
        return segs

    # 1) 先从低分片段上收回「对齐时多扩出来」的部分（先收尾、再收头）
    for i in sorted(range(len(segs)), key=lambda k: segs[k].get("score", 5)):
        if total <= target_max:
            break
        seg = segs[i]

        floor_end = max(seg.get("orig_end", seg["start"] + min_clip),
                        seg["start"] + min_clip)
        shrink = min(total - target_max, max(0.0, seg["end"] - floor_end))
        if shrink > 0:
            seg["end"] -= shrink
            total -= shrink
        if total <= target_max:
            break

        ceil_start = min(seg.get("orig_start", seg["end"] - min_clip),
                         seg["end"] - min_clip)
        shrink = min(total - target_max, max(0.0, ceil_start - seg["start"]))
        if shrink > 0:
            seg["start"] += shrink
            total -= shrink

    # 2) 仍然超长：整段删掉低分片段，但不能低于目标下限
    while total > target_max and len(segs) > 1:
        i = min(range(len(segs)), key=lambda k: segs[k].get("score", 5))
        dur = segs[i]["end"] - segs[i]["start"]
        if total - dur < target_min:
            break
        total -= dur
        segs.pop(i)

    segs.sort(key=lambda x: (x["start"], x["end"]))
    return segs


# ---------------------------------------------------------------------------
# 重编码（修复 Premiere 时间轴错位）
# ---------------------------------------------------------------------------
def probe_streams(path):
    """用 ffprobe 读出时长、帧率、码率等参数；缺失的字段返回 None / 0。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True)
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    fmt = info.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    duration = num(fmt.get("duration"))
    if not duration and video:
        duration = num(video.get("duration"))

    fps = fps_raw = None
    if video:
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = str(video.get(key) or "")
            if "/" in raw:
                n, _, d = raw.partition("/")
                n, d = num(n), num(d)
                if n and d and n / d > 1:
                    fps, fps_raw = n / d, raw
                    break
            else:
                f = num(raw)
                if f and f > 1:
                    fps, fps_raw = f, raw
                    break

    v_bitrate = num(video.get("bit_rate")) if video else None
    a_bitrate = num(audio.get("bit_rate")) if audio else None
    if not v_bitrate:
        total = num(fmt.get("bit_rate"))
        if total:
            v_bitrate = max(0.0, total - (a_bitrate or 0.0))

    return {
        "duration": duration or 0.0,
        "fps": fps,
        "fps_raw": fps_raw,
        "v_bitrate": v_bitrate or 0.0,
        "a_bitrate": a_bitrate or 0.0,
        "a_channels": int(num(audio.get("channels")) or 0) if audio else 0,
        "a_rate": int(num(audio.get("sample_rate")) or 0) if audio else 0,
        "has_audio": audio is not None,
        "width": int(num(video.get("width")) or 0) if video else 0,
        "height": int(num(video.get("height")) or 0) if video else 0,
    }


_STD_FPS = [
    (24000 / 1001, "24000/1001"), (24, "24"), (25, "25"),
    (30000 / 1001, "30000/1001"), (30, "30"), (48, "48"),
    (48000 / 1001, "48000/1001"), (50, "50"),
    (60000 / 1001, "60000/1001"), (60, "60"),
    (100, "100"), (120000 / 1001, "120000/1001"), (120, "120"), (144, "144"),
]


def normalize_fps(fps):
    """把探测到的帧率吸附到标准帧率。

    拼接后的成片是 VFR，ffprobe 的 avg_frame_rate 可能给出
    「1425008000/47580349」这种怪值；直接用会让成片帧率不标准，
    所以和标准帧率相差 1% 以内的都当成标准帧率，否则保留 3 位小数。
    """
    if not fps or fps <= 1:
        return None
    best_value, best_expr = min(_STD_FPS, key=lambda kv: abs(kv[0] - fps))
    if abs(best_value - fps) <= best_value * 0.01:
        return best_expr
    return f"{fps:.3f}"


def reencode_cfr(src, dst, crf=REENCODE_CRF, preset=REENCODE_PRESET,
                 maxrate_ratio=REENCODE_MAXRATE_RATIO,
                 audio_bitrate=REENCODE_AUDIO_BITRATE):
    """把 stream copy 拼出来的成片整体重编码成恒定帧率（CFR）。

    为什么要重编码：
        stream copy 只能从关键帧开始切，拼接后视频是 VFR（帧间隔不均匀）、
        时间戳是拼接式的（每段都从 0 重新计时），而且音频 / 视频的 time base
        也不统一。Premiere 读这种文件会按平均帧率推算帧位置，成片越往后
        错位越明显。整体重编码后时间戳连续、帧率恒定，导入就不会错位。

    为什么画质 / 体积基本不变：
        - x264 CRF 18：视觉上和源片基本看不出差别；
        - VBV 上限 = 源视频码率 × 2（bufsize 取 2 倍），防止重编码后体积膨胀；
        - 音频按源码率重新编码（夹在 96k~320k 之间），声道 / 采样率沿用源片；
        - 分辨率沿用源片，像素格式统一成 yuv420p（兼容性最好）。
    """
    info = probe_streams(src)

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-fflags", "+genpts",
        "-i", src,
        "-map", "0:v:0",
    ]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0"]

    cmd += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", f"{float(crf):g}",
        "-pix_fmt", "yuv420p",
    ]
    # 恒定帧率：显式指定帧率 + CFR 模式，缺帧补帧、多帧丢弃
    cmd += ["-fps_mode", "cfr"]
    fps_expr = normalize_fps(info["fps"])
    if fps_expr:
        cmd += ["-r", fps_expr]
    if info["v_bitrate"] > 0:
        cap = int(info["v_bitrate"] * float(maxrate_ratio))
        cmd += ["-maxrate", str(cap), "-bufsize", str(cap * 2)]

    if info["has_audio"]:
        abr = int(audio_bitrate or info["a_bitrate"] or 0)
        abr = min(max(abr or 128000, 96000), 320000)
        cmd += ["-c:a", "aac", "-b:a", str(abr)]
        if info["a_rate"]:
            cmd += ["-ar", str(info["a_rate"])]
        if info["a_channels"]:
            cmd += ["-ac", str(info["a_channels"])]
        # stream copy 拼出来的音频可能有空隙，async 重采样把它们补齐，避免音画漂移
        cmd += ["-af", "aresample=async=1:first_pts=0"]

    cmd += ["-avoid_negative_ts", "make_zero"]
    if dst.lower().endswith((".mp4", ".m4v", ".mov")):
        cmd += ["-movflags", "+faststart"]
    cmd.append(dst)

    log(f"  重编码命令: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"重编码失败（退出码 {proc.returncode}）: {dst}")


def rescale_timeline(timeline, raw_path, out_path):
    """重编码后总时长可能有毫秒级差异，等比修正成片时间轴，保证字幕不错位。"""
    raw_dur = probe_duration(raw_path)
    out_dur = probe_duration(out_path)
    if raw_dur <= 0 or out_dur <= 0 or abs(out_dur - raw_dur) < 0.5:
        return timeline
    k = out_dur / raw_dur
    log(f"  重编码后时长 {raw_dur:.3f}s -> {out_dur:.3f}s，成片时间轴 × {k:.6f}")
    fixed = []
    for item in timeline:
        item = dict(item)
        item["out_start"] *= k
        item["actual"] *= k
        item["shift"] = max(0.0, item["actual"] - float(item["length"]))
        fixed.append(item)
    return fixed


def probe_signature(path):
    """探测「能不能直接 concat」的码流参数（编码 / 分辨率 / 像素格式 / 音频）。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True)
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video:
        raise SystemExit(f"没有视频流: {path}")

    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = str(video.get(key) or "")
        if "/" not in raw:
            continue
        n, _, d = raw.partition("/")
        try:
            n, d = float(n), float(d)
        except ValueError:
            continue
        if d and n / d > 1:
            fps = n / d
            break

    return {
        "video_key": (video.get("codec_name"), video.get("profile"),
                      int(video.get("width") or 0),
                      int(video.get("height") or 0), video.get("pix_fmt")),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps_expr": normalize_fps(fps),
        "has_audio": audio is not None,
        "a_rate": int(audio.get("sample_rate") or 0) if audio else 0,
        "a_channels": int(audio.get("channels") or 0) if audio else 0,
    }


def parts_signatures(parts):
    """每个分片的码流参数，供 render() 判断能否 stream copy 拼接。"""
    return [(p["name"], probe_signature(p["path"])) for p in parts]


def cut_segment_reencoded(src, start, duration, dst, target, src_info=None,
                          crf=REENCODE_CRF, preset=REENCODE_PRESET,
                          maxrate_ratio=REENCODE_MAXRATE_RATIO,
                          audio_bitrate=REENCODE_AUDIO_BITRATE):
    """帧精确地切一段，并统一重编码成 H.264/AAC（不同源也能安全 concat）。

    -ss 放在 -i 之前 + 重编码 = 精确定位（解码后丢弃多余帧），
    不会像 stream copy 那样「从关键帧起切、多带一个 GOP」。
    target 是第一个分片的参数（宽高 / 帧率 / 音频），所有段按它编码，
    这样 concat -c copy 才是合法的单条码流。
    """
    info = src_info or probe_streams(src)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", src,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", preset,
        "-crf", f"{float(crf):g}", "-pix_fmt", "yuv420p",
    ]
    if (info.get("width"), info.get("height")) != (target["width"],
                                                    target["height"]):
        cmd += ["-vf", f"scale={target['width']}:{target['height']}"]
    cmd += ["-fps_mode", "cfr"]
    if target.get("fps_expr"):
        cmd += ["-r", str(target["fps_expr"])]
    v_bitrate = info.get("v_bitrate") or 0
    if v_bitrate > 0:
        cap = int(v_bitrate * float(maxrate_ratio))
        cmd += ["-maxrate", str(cap), "-bufsize", str(cap * 2)]

    if target.get("has_audio") and info.get("has_audio"):
        abr = int(audio_bitrate or info.get("a_bitrate") or 0) or 128000
        abr = min(max(abr, 96000), 320000)
        cmd += ["-map", "0:a:0",
                "-c:a", "aac", "-b:a", str(abr)]
        if target.get("a_rate"):
            cmd += ["-ar", str(target["a_rate"])]
        if target.get("a_channels"):
            cmd += ["-ac", str(target["a_channels"])]
        # 补齐音频空隙，避免拼接后音画漂移
        cmd += ["-af", "aresample=async=1:first_pts=0"]

    cmd += ["-avoid_negative_ts", "make_zero", dst]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"重编码分片失败（退出码 {proc.returncode}）: {src}")


# ---------------------------------------------------------------------------
# 剪辑（复用 auto_edit 的 stream copy 逻辑）
# ---------------------------------------------------------------------------
def render(parts, segments, out_path, keep_temp=False,
           reencode=REENCODE_OUTPUT, crf=REENCODE_CRF):
    """按全局区间逐段切（跨分片自动拆），再依次拼接，最后整体重编码成 CFR。

    返回 timeline：每段在实际成片里的位置，供生成对齐字幕用。
    stream copy 只能从关键帧开始切，实际起点可能比请求的早一个 GOP，
    所以这里逐段记录「实际时长 / 实际提前量」，字幕才能跟着对齐。
    """
    tmp_dir = tempfile.mkdtemp(prefix="edit_caption_")
    timeline = []
    try:
        # 分片码流不一致（典型：HEVC 与 H.264 混用）时，stream copy 的 concat
        # 会把两种码流写进同一条轨道，解码器从第一个异构片段开始就整段解不出来，
        # 播放器 / Premiere 只能一直冻在上一帧（实测能冻十几分钟）。
        sigs = parts_signatures(parts)
        mixed = len({sig["video_key"] for _, sig in sigs}) > 1
        target, src_info = None, {}
        if mixed:
            kinds = {}
            for _, sig in sigs:
                kinds[sig["video_key"][0]] = kinds.get(sig["video_key"][0], 0) + 1
            log("  ! 分片编码不一致：" +
                "，".join(f"{k}×{v} 个" for k, v in kinds.items()))
            log("    不同编码不能 stream copy 拼接（H.264 塞进 HEVC 轨道会让解码器"
                "整段失效，画面长时间冻结）")
            log("    -> 逐段重编码成统一 H.264/AAC 后再无损拼接（帧精确，且直接是 CFR）")
            first = sigs[0][1]
            target = {
                "width": first["width"] or 1920,
                "height": first["height"] or 1080,
                "fps_expr": first["fps_expr"] or "30",
                "has_audio": all(sig["has_audio"] for _, sig in sigs),
                "a_rate": first["a_rate"],
                "a_channels": first["a_channels"],
            }
            for p in parts:
                src_info[p["path"]] = probe_streams(p["path"])
            if not target["has_audio"]:
                log("    ! 有分片没有音轨，成片将不保留音频")

        outs = []
        n = 0
        cursor = 0.0
        for i, seg in enumerate(segments, 1):
            pieces = map_segment_to_parts(parts, seg["start"], seg["end"])
            if not pieces:
                continue
            note = "" if len(pieces) == 1 else f"（跨 {len(pieces)} 个分片）"
            log(f"  [{i}/{len(segments)}] {fmt_hms(seg['start'])} ~ "
                f"{fmt_hms(seg['end'])} ({seg['end'] - seg['start']:.1f}s) "
                f"{seg.get('title', '')}{note}")
            for part, ls, le in pieces:
                n += 1
                ext = os.path.splitext(part["path"])[1].lower() or ".mp4"
                dst = os.path.join(tmp_dir, f"seg_{n:04d}{ext}")
                if len(pieces) > 1:
                    log(f"      {part['name']}  {fmt_hms(ls)} ~ {fmt_hms(le)}"
                        f" ({le - ls:.1f}s)")
                if mixed:
                    cut_segment_reencoded(part["path"], ls, le - ls, dst, target,
                                          src_info.get(part["path"]), crf=crf)
                else:
                    cut_segment(part["path"], ls, le - ls, dst)
                outs.append(dst)

                actual = probe_duration(dst)
                timeline.append({
                    "seg": i,
                    "global_start": part["offset"] + ls,
                    "length": le - ls,
                    "actual": actual,
                    "shift": max(0.0, actual - (le - ls)),
                    "out_start": cursor,
                })
                cursor += actual

        list_path = os.path.join(tmp_dir, "segments.txt")
        if mixed:
            # 每段已经是统一的 CFR H.264/AAC，直接无损拼接，不需要再整体重编码
            log(f"拼接 {len(outs)} 段 -> {out_path}")
            concat_segments(outs, out_path, list_path)
        elif reencode:
            ext = os.path.splitext(out_path)[1].lower() or ".mp4"
            raw = os.path.join(tmp_dir, "concat_raw" + ext)
            log(f"拼接 {len(outs)} 段 -> 临时文件（随后整体重编码）")
            concat_segments(outs, raw, list_path)
            log("整体重编码（恒定帧率 CFR，修复 Premiere 时间轴错位）...")
            t0 = time.time()
            reencode_cfr(raw, out_path, crf=crf)
            log(f"  重编码完成: {out_path}（耗时 {time.time() - t0:.1f}s）")
            timeline = rescale_timeline(timeline, raw, out_path)
        else:
            log(f"拼接 {len(outs)} 段 -> {out_path}")
            concat_segments(outs, out_path, list_path)
    finally:
        if keep_temp:
            log(f"临时分片保留在: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return timeline


def load_plan(path, duration=None):
    """读取 save_plan() 写出的 .plan.json，返回片段列表。

    直接按计划渲染时用：跳过 AI 分析与挑选，剪辑点原样使用。
    给了 duration 就检查计划是否还落在当前素材的时间轴内
    （素材换过/少了分片时时间轴会变短，必须报错而不是剪出错片）。
    """
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"读取剪辑计划失败: {path}（{e}）")
    if isinstance(data, dict):                 # 兼容以后带元信息的格式
        data = data.get("segments") or []
    if not isinstance(data, list):
        raise SystemExit(f"剪辑计划格式不对（应为片段数组）: {path}")

    segments = []
    for item in data:
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (TypeError, ValueError):
            continue
        if end - start <= 0:
            continue
        seg = dict(item)
        seg["start"], seg["end"] = start, end
        segments.append(seg)
    segments.sort(key=lambda s: s["start"])
    if not segments:
        raise SystemExit(f"剪辑计划里没有可用片段: {path}")

    if duration is not None:
        latest = max(s["end"] for s in segments)
        if latest > duration + 1.0:
            raise SystemExit(
                f"剪辑计划与当前素材不匹配：计划最晚到 {fmt_hms(latest)}，"
                f"但当前时间轴只有 {fmt_hms(duration)}。\n"
                f"（素材可能换过或缺了分片；确认无误可加 --no-plan 重新分析）")
    return segments


def save_plan(segments, out_path, parts=None):
    base = os.path.splitext(out_path)[0]
    with open(base + ".plan.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    with open(base + ".plan.txt", "w", encoding="utf-8", newline="\n") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i:02d}  {fmt_hms(seg['start'])} ~ {fmt_hms(seg['end'])}"
                    f"  ({seg['end'] - seg['start']:.1f}s)"
                    f"  score={seg.get('score', 0):.0f}\n")
            if seg.get("title"):
                f.write(f"    标题: {seg['title']}\n")
            if seg.get("reason"):
                f.write(f"    理由: {seg['reason']}\n")
            if seg.get("text"):
                f.write(f"    台词: {seg['text']}\n")
            if seg.get("type"):
                f.write(f"    类型: {seg['type']}\n")
            if seg.get("streamer_ratio") is not None:
                f.write(f"    主播说话: {seg['streamer_ratio'] * 100:.0f}%\n")
            if seg.get("ctx_sec"):
                f.write(f"    上文补齐: {seg['ctx_sec']:+.1f}s"
                        f"（{seg.get('ctx_reason', '')}）\n")
            if seg.get("orig_start") is not None:
                ds = seg["start"] - seg["orig_start"]
                de = seg["end"] - seg.get("orig_end", seg["end"])
                if abs(ds) > 0.05 or abs(de) > 0.05:
                    f.write(f"    音频对齐: 起点 {ds:+.1f}s, 终点 {de:+.1f}s\n")
            if parts and len(parts) > 1:
                for part, ls, le in map_segment_to_parts(
                        parts, seg["start"], seg["end"]):
                    f.write(f"    来源: {part['name']} "
                            f"{fmt_hms(ls)} ~ {fmt_hms(le)}\n")
            f.write("\n")
    log(f"剪辑计划: {base}.plan.txt")


# ---------------------------------------------------------------------------
# 成片字幕：把全局时间轴的字幕搬到成片时间轴（片段拼接后的位置）
# ---------------------------------------------------------------------------
def fmt_srt_time(seconds):
    """秒 -> SRT 时间戳 HH:MM:SS,mmm。"""
    ms = max(int(round(float(seconds) * 1000)), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_output_subtitles(entries, timeline, min_dur=MIN_OUT_SUB_SEC):
    """按实际剪辑结果，把字幕重映射到成片时间轴。

    timeline 来自 render()：每段记录了请求的源区间、实际提前量、成片起点。
    跨边界的字幕会被裁到片段内；裁完不足 min_dur 的（切点戈到的半句）丢掉。
    """
    out = []
    for piece in timeline:
        src_lo = float(piece["global_start"])
        src_hi = src_lo + float(piece["length"])
        # 成片里这一段的内容从 out_start 开始，但前 shift 秒是 keyframe 多带的前文，
        # 所以源时间 src_lo 实际对应成片时间 out_start + shift
        out_lo = float(piece["out_start"]) + float(piece.get("shift", 0.0))
        for e in entries:
            s = max(float(e["start"]), src_lo)
            t = min(float(e["end"]), src_hi)
            if t - s < min_dur:
                continue
            text = (e.get("text") or "").strip()
            if not text:
                continue
            out.append({"start": out_lo + (s - src_lo),
                        "end": out_lo + (t - src_lo),
                        "text": text})

    out.sort(key=lambda x: (x["start"], x["end"]))
    for i in range(1, len(out)):          # 保险：同一时间轴上不要重叠
        if out[i]["start"] < out[i - 1]["end"]:
            out[i]["start"] = out[i - 1]["end"]
        if out[i]["end"] - out[i]["start"] < min_dur:
            out[i]["end"] = out[i]["start"] + min_dur
    return out


def write_srt_file(subtitles, path):
    """写标准 SRT（UTF-8）。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, s in enumerate(subtitles, 1):
            f.write(f"{i}\n{fmt_srt_time(s['start'])} --> "
                    f"{fmt_srt_time(s['end'])}\n{s['text']}\n\n")
    return len(subtitles)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(
        description="用字幕（可选弹幕）交给大模型挑选精华片段并顺序剪辑")
    parser.add_argument("--video", default=None, help="输入视频（单文件模式）")
    parser.add_argument("--video-dir", default=None,
                        help="组合剪辑：文件夹里的多个视频按字母顺序当成分段拼接"
                             f"（不传 --video 时默认 {VIDEO_DIR}）")
    parser.add_argument("--srt", default=None,
                        help="字幕 SRT；传文件夹时按分片一一配对")
    parser.add_argument("--danmaku", default=None,
                        help="弹幕 XML；传文件夹时按分片一一配对")
    parser.add_argument("--use-danmaku", dest="use_danmaku",
                        action="store_true", default=USE_DANMAKU,
                        help="分析字幕时叠加弹幕热度")
    parser.add_argument("--no-danmaku", dest="use_danmaku",
                        action="store_false", help="不使用弹幕")
    parser.add_argument("-o", "--output", default=OUTPUT_PATH,
                        help="输出视频路径，默认 <源视频文件夹>/export/<名字>_edited.mp4")
    parser.add_argument("--start", type=float, default=VIDEO_START,
                        help=f"全局时间轴起点秒数，默认 {VIDEO_START:g}")
    parser.add_argument("--end", type=float, default=VIDEO_END,
                        help="全局时间轴终点秒数，默认到结尾")
    parser.add_argument("--target-min", type=float, default=TARGET_MIN_SEC,
                        help=f"成片最短时长秒，默认 {TARGET_MIN_SEC:g}")
    parser.add_argument("--target-max", type=float, default=TARGET_MAX_SEC,
                        help=f"成片最长时长秒，默认 {TARGET_MAX_SEC:g}")
    parser.add_argument("--min-clip", type=float, default=MIN_CLIP_SEC,
                        help=f"单段最短时长秒，默认 {MIN_CLIP_SEC:g}")
    parser.add_argument("--prompt", default=EDIT_PROMPT,
                        help="剪辑方向/提示词，例如「眼神」")
    parser.add_argument("--streamer-target", type=float,
                        default=STREAMER_TARGET_RATIO,
                        help=f"成片里主播说话占语音的目标比例 0-1，默认 "
                             f"{STREAMER_TARGET_RATIO:g}（0 = 不约束）")
    parser.add_argument("--min-clip-streamer", type=float,
                        default=MIN_CLIP_STREAMER_RATIO,
                        help=f"候选里主播说话占比低于此值就不选，默认 "
                             f"{MIN_CLIP_STREAMER_RATIO:g}（0 = 不过滤）")
    parser.add_argument("--no-mix", dest="mix", action="store_false",
                        default=CONTENT_MIX_ENABLED,
                        help="关闭内容配比校正（只按分数 / 时长挑）")
    parser.add_argument("--no-trim-streamer", dest="trim_streamer",
                        action="store_false", default=TRIM_TO_STREAMER,
                        help="不裁片段首尾的大段非主播语音")
    parser.add_argument("--plan", default=None,
                        help="已有剪辑计划 .plan.json：直接按它渲染，不再调用大模型")
    parser.add_argument("--chunk-sec", type=float, default=LLM_CHUNK_SEC,
                        help=f"每块字幕时长秒，默认 {LLM_CHUNK_SEC:g}")
    parser.add_argument("--final-candidates", type=int,
                        default=LLM_FINAL_MAX_CANDIDATES,
                        help=f"全局挑选最多喂给大模型多少候选，"
                             f"默认 {LLM_FINAL_MAX_CANDIDATES}")
    parser.add_argument("--context-sec", type=float, default=LLM_CONTEXT_SEC,
                        help=f"每块前后附带的上下文秒数，默认 {LLM_CONTEXT_SEC:g}")
    parser.add_argument("--snap-shift", type=float, default=SNAP_MAX_SHIFT,
                        help=f"剪辑点向外找静音时最多扩的秒数，默认 {SNAP_MAX_SHIFT:g}")
    parser.add_argument("--no-snap", action="store_true",
                        help="关闭剪辑点音频对齐（不向外扩边界）")
    # 兼容旧命令行：--no-correct / --correct 已不起作用（纠错搬到 sensevoice_srt.py）
    parser.add_argument("--no-correct", dest="correct", action="store_false",
                        default=None, help=argparse.SUPPRESS)
    parser.add_argument("--correct", dest="correct", action="store_true",
                        default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default=LLM_MODEL, help="大模型名称")
    parser.add_argument("--srt-out", default=None,
                        help="成片字幕路径，默认 <成片>.srt（与成片同在 export 里）")
    parser.add_argument("--no-output-srt", dest="write_srt",
                        action="store_false", default=WRITE_OUTPUT_SRT,
                        help="不生成对齐到成片的字幕")
    parser.add_argument("--dry-run", action="store_true",
                        help="只分析并输出计划，不渲染视频")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留临时分片")
    parser.add_argument("--no-reencode", dest="reencode", action="store_false",
                        default=REENCODE_OUTPUT,
                        help="拼接后不再整体重编码（保留 stream copy 结果；"
                             "时间戳可能让 Premiere 时间错位）")
    parser.add_argument("--reencode-crf", type=float, default=REENCODE_CRF,
                        help=f"重编码画质档位，默认 {REENCODE_CRF:g}")
    args = parser.parse_args()

    if not SNAP_BOUNDARIES:
        args.no_snap = True
    LLM_MODEL = args.model

    PROFILE["mix"] = bool(args.mix)
    PROFILE["streamer_target"] = max(0.0, min(float(args.streamer_target), 1.0))
    PROFILE["min_clip_streamer"] = max(0.0, min(float(args.min_clip_streamer), 1.0))
    PROFILE["trim_streamer"] = bool(args.trim_streamer)

    require_tools()

    if args.video and args.video_dir:
        raise SystemExit("--video 与 --video-dir 只能二选一")
    if not args.video and not args.video_dir:
        args.video_dir = VIDEO_DIR
    if args.srt is None:
        if args.video:                      # 单文件模式：优先用同名 SRT
            guess = os.path.splitext(args.video)[0] + ".srt"
            args.srt = guess if os.path.isfile(guess) else SRT_PATH
        else:
            args.srt = args.video_dir
    if args.danmaku is None:
        args.danmaku = args.video_dir if args.video_dir else DANMAKU_PATH

    # ---- 组装时间轴（单文件 = 只有一个分片）----
    if args.video_dir:
        if not os.path.isdir(args.video_dir):
            raise SystemExit(f"找不到视频文件夹: {args.video_dir}")
        videos = list_files(args.video_dir, VIDEO_EXTS)
        if not videos:
            raise SystemExit(f"文件夹里没有可用的视频: {args.video_dir}")
    else:
        if not os.path.isfile(args.video):
            raise SystemExit(f"找不到视频文件: {args.video}")
        videos = [args.video]

    parts = build_parts(videos)
    duration = timeline_duration(parts)
    srt_mode, srt_source = resolve_aux_source(args.srt, SRT_EXTS)
    if srt_mode == "none" and not args.plan:
        raise SystemExit(f"找不到字幕文件: {args.srt}")

    if args.target_min > args.target_max:
        raise SystemExit("成片最短时长不能大于最长时长")
    if args.min_clip <= 0:
        raise SystemExit("单段最短时长必须大于 0")
    if not LLM_API_KEY and not args.plan:
        raise SystemExit(
            "未设置 DEEPSEEK_API_KEY。PowerShell 示例:\n"
            '    $env:DEEPSEEK_API_KEY="sk-xxxx"')

    v_start = max(0.0, args.start)
    v_end = duration if args.end is None else min(duration, args.end)
    if v_end - v_start < args.min_clip:
        raise SystemExit("可用的视频区间太短")

    # ---- 已有剪辑计划：跳过 AI，直接按计划渲染 ----
    plan_segments, plan_path = None, None
    if args.plan:
        plan_path = os.path.abspath(args.plan)
        if not os.path.isfile(plan_path):
            raise SystemExit(f"找不到剪辑计划: {args.plan}")
        plan_segments = load_plan(plan_path, duration)
        if (min(s["start"] for s in plan_segments) < v_start
                or max(s["end"] for s in plan_segments) > v_end):
            log(f"  ! 计划区间超出 --start/--end（{fmt_hms(v_start)} ~ "
                f"{fmt_hms(v_end)}），仍按计划原样剪辑")

    out_path = args.output
    if not out_path:
        if args.video_dir:
            root = os.path.abspath(args.video_dir)
            base = os.path.basename(os.path.normpath(root))
            out_path = os.path.join(root, EXPORT_DIR_NAME, base + "_edited.mp4")
        else:
            video = os.path.abspath(args.video)
            base = os.path.splitext(os.path.basename(video))[0]
            out_path = os.path.join(os.path.dirname(video), EXPORT_DIR_NAME,
                                    base + "_edited.mp4")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if len(parts) > 1:
        log(f"分段     : {len(parts)} 个（按文件名顺序拼成全局时间轴）")
        for p in parts:
            log(f"  {p['index'] + 1:02d} {p['name']}  "
                f"{fmt_hms(p['duration'])} -> 全局 {fmt_hms(p['offset'])}")
    else:
        log(f"视频     : {parts[0]['path']}")
    log(f"时长     : {fmt_hms(duration)}，使用 {fmt_hms(v_start)} ~ {fmt_hms(v_end)}")
    log(f"输出     : {out_path}")
    log(f"字幕     : {srt_mode}（{args.srt}）")
    log(f"弹幕     : {'开' if args.use_danmaku else '关'}（{args.danmaku}）")
    log("音频对齐 : " + (f"开（边界向外扩 ≤ {args.snap_shift:g}s 找静音）"
                          if not args.no_snap and args.snap_shift > 0 else "关"))
    if plan_segments:
        log(f"剪辑计划 : {plan_path}（{len(plan_segments)} 段，跳过 AI 分析）")
    log(f"目标成片 : {fmt_hms(args.target_min)} ~ {fmt_hms(args.target_max)}"
        f"，单段 >= {args.min_clip:g}s")
    log(f"提示词   : {args.prompt or '（空）'}")
    if not plan_segments:
        log("剪辑倾向 : " + (f"内容配比 {_mix_range_text()}" if PROFILE["mix"]
                             else "不校正内容配比"))

    # ---- 读字幕 ----
    entries = load_entries(parts, srt_mode, srt_source, v_start, v_end)
    if not entries:
        if plan_segments:
            log("  ! 没有读到字幕：仍按计划剪辑，但不会生成成片字幕")
            args.write_srt = False
        else:
            raise SystemExit("字幕在指定区间内没有内容")
    log(f"字幕条数 : {len(entries)}")
    PROFILE["has_streamer"] = any(_speaker_entry(e) for e in entries)
    if PROFILE["has_streamer"]:
        log(f"主播标记 : 有（候选需 >= {PROFILE['min_clip_streamer'] * 100:.0f}% 主播说话，"
            f"成片目标 {PROFILE['streamer_target'] * 100:.0f}%）")
    else:
        log("主播标记 : 无（字幕没有（主播）标记，跳过主播相关筛选）")

    # ---- 读弹幕（可选）----
    buckets = {}
    if args.use_danmaku and not plan_segments:
        danmaku_mode, danmaku_source = resolve_aux_source(args.danmaku,
                                                          DANMAKU_EXTS)
        if danmaku_mode == "none":
            log(f"  ! 找不到弹幕文件，忽略: {args.danmaku}")
        else:
            items = load_danmaku(parts, danmaku_mode, danmaku_source)
            buckets = build_danmaku_buckets(items)
            log(f"弹幕条数 : {len(items)}，聚合桶 {len(buckets)} 个")

    # ---- 分块（只有 AI 分析用得到；按计划渲染时跳过）----
    chunks = []
    if not plan_segments:
        chunks = build_chunks(entries, args.chunk_sec)
        log(f"字幕分块 : {len(chunks)} 块（每块 {args.chunk_sec:g}s）")

    if plan_segments:
        # ---- 直接使用现成计划：不做分块分析 / 全局挑选 ----
        log(f"按计划剪辑 : {len(plan_segments)} 段，跳过 AI 分析与挑选")
        segments = plan_segments
    else:
        # ---- 分块分析 ----
        log("")
        candidates, cid = [], 0
        for ci, (lo, hi, chunk_entries) in enumerate(chunks, 1):
            lo = max(lo, v_start)
            hi = min(hi, v_end)
            if hi - lo <= 0 or not chunk_entries:
                continue

            danmaku_text = (format_danmaku(buckets, lo, hi)
                            if args.use_danmaku else "（未启用弹幕）")
            log(f"[{ci}/{len(chunks)}] 分析 {fmt_hms(lo)} ~ {fmt_hms(hi)}"
                f"（字幕 {len(chunk_entries)} 条）")

            found = analyze_chunk(entries, lo, hi, danmaku_text,
                                  args.min_clip, LLM_MAX_CANDIDATES_PER_CHUNK,
                                  args.prompt, args.context_sec)
            for item in found:
                cid += 1
                item["id"] = cid
                candidates.append(item)
            log(f"    -> 候选 {len(found)} 个")

        if not candidates:
            raise SystemExit("没有获得任何候选片段，退出")

        log(f"\n候选合计 : {len(candidates)} 个")
        pool = shortlist_candidates(candidates, args.final_candidates)
        if len(pool) < len(candidates):
            log(f"候选精简 : {len(pool)} 个（截断以控制提示词长度）")

        # ---- 全局挑选 ----
        log("全局挑选 ...")
        chosen = select_with_llm(pool, entries, args.target_min,
                                 args.target_max, args.min_clip)
        if not chosen:
            chosen = sorted(pool, key=lambda x: -x["score"])[:20]
        segments = fit_selection(pool, chosen, args.target_min,
                                 args.target_max, args.min_clip, v_start, v_end)
        if not segments:
            raise SystemExit("全局校验后没有可用片段，退出")

        # ---- 倾向校正：主播说话占比 + 内容配比（向手动剪辑样本靠拢）----
        before = segment_profile(entries, segments)
        segments = align_to_profile(segments, pool, args.target_min,
                                    args.target_max, args.min_clip)
        after = segment_profile(entries, segments)
        log(f"剪辑倾向 : 主播说话 {before['streamer'] * 100:.0f}% -> "
            f"{after['streamer'] * 100:.0f}%（目标 {PROFILE['streamer_target'] * 100:.0f}%）")
        log(f"内容配比 : {_fmt_mix(after['mix'])}")

        # ---- 逻辑连接：主播吐槽 / 评论的上文出处一起带上（关联强才补）----
        segments, link = stitch_context(
            segments, entries, args.target_max + OVERFLOW_TOLERANCE_SEC)
        if link["extended"]:
            log(f"逻辑连接 : {link['extended']} 段往前补上了出处"
                f"（+{link['extra']:.1f}s）")
        else:
            log("逻辑连接 : 没有需要往前补的强关联上文")

    # ---- 音频对齐：边界向外扩到静音处，保证句子 / 剧情完整 ----
    if plan_segments:
        log("音频对齐 : 跳过（计划里的剪辑点已经是最终位置；要重调请重新分析）")
    elif not args.no_snap and args.snap_shift > 0:
        segments, moved = snap_segments(parts, segments, args.snap_shift,
                                        v_start, v_end)
        log(f"音频对齐 : {moved}/{len(segments)} 段边界向外扩动过")
        before = len(segments)
        segments = merge_segments(segments)
        if len(segments) != before:
            log(f"合并片段 : {before} 段 -> {len(segments)} 段（外扩后重叠）")
        segments = enforce_duration(segments, args.target_min,
                                    args.target_max + OVERFLOW_TOLERANCE_SEC,
                                    args.min_clip)

    total = sum(s["end"] - s["start"] for s in segments)
    if not plan_segments and total > args.target_max + 0.5:
        log(f"  ! 对齐后总时长 {total:.1f}s 超出上限 {args.target_max:g}s，"
            f"在 {OVERFLOW_TOLERANCE_SEC:g}s 容差内优先保证片段完整")
    log(f"最终片段 : {len(segments)} 段，合计 {fmt_hms(total)}（{total:.1f}s）")
    if PROFILE["has_streamer"]:
        final = segment_profile(entries, segments)
        log(f"成片倾向 : 主播说话 {final['streamer'] * 100:.0f}%"
            f"（目标 {PROFILE['streamer_target'] * 100:.0f}%），"
            f"内容 {_fmt_mix(final['mix'])}")
    for i, seg in enumerate(segments, 1):
        log(f"  {i:02d} {fmt_hms(seg['start'])} ~ {fmt_hms(seg['end'])} "
            f"({seg['end'] - seg['start']:.1f}s) "
            f"score={seg.get('score', 0):.0f} "
            f"[{seg.get('type') or '其他'} "
            f"主播{(seg.get('streamer_ratio') or 0) * 100:.0f}%] "
            f"{seg.get('title', '')}")

    if plan_segments:
        log(f"沿用剪辑计划: {plan_path}（不重写）")
    else:
        save_plan(segments, out_path, parts)

    if args.dry_run:
        log("dry-run：不渲染视频")
        return

    src_ext = os.path.splitext(parts[0]["path"])[1].lower()
    if os.path.splitext(out_path)[1].lower() != src_ext:
        out_path = os.path.splitext(out_path)[0] + src_ext
        log(f"输出容器改为与源一致: {out_path}")

    t0 = time.time()
    timeline = render(parts, segments, out_path, args.keep_temp,
                      reencode=args.reencode, crf=args.reencode_crf)
    log(f"完成: {out_path}（耗时 {time.time() - t0:.1f}s）")

    # ---- 成片字幕：时间轴按实际剪辑结果重算，文本用（已在 ASR 阶段纠错过的）版本 ----
    if args.write_srt:
        srt_out = args.srt_out or (os.path.splitext(out_path)[0] + ".srt")
        subtitles = build_output_subtitles(entries, timeline)
        if subtitles:
            srt_out = os.path.abspath(srt_out)
            count = write_srt_file(subtitles, srt_out)
            shift = max((p.get("shift", 0.0) for p in timeline), default=0.0)
            log(f"成片字幕: {srt_out}（{count} 条，已对齐到成片时间轴"
                f"{'，关键帧最多提前 ' + format(shift, '.1f') + 's 已补偿' if shift > 0.1 else ''}）")
        else:
            log("成片字幕: 没有可用的字幕，跳过")


if __name__ == "__main__":
    main()

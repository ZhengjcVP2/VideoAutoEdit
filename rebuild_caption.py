#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_caption.py — 把旧字幕重新对到新视频上，并按剪辑计划重生成新时间点的目录

场景:
    手里有一条剪过的视频（可能剪掉了几秒、也可能剪掉一整块，还可能在开头
    加了一段由视频自己内容拼成的片头），以及原始视频的字幕文件。正文顺序
    不变、基本没有新内容，只需要把旧字幕重新对到新视频的时间轴上。

做法（不依赖源视频，也不依赖剪辑计划）:
    1. 先拿到「新视频的语音时间轴」：默认调 sensevoice_srt.py 对新视频识别一遍
       （结果缓存在 <视频目录>/rebuild_cache/，重跑直接复用；也可以用 --asr
       直接喂一份现成的 SRT / asr.jsonl）；
    2. 旧字幕文本和新视频 ASR 文本都归一化成纯文字序列（去标点、去（主播）
       这类标注），用 8 字 n-gram 做锚点：新文本里每个 n-gram 在旧文本里找到
       所有出现位置，就得到一对（新时间, 旧时间）；
    3. 按 2s 时间桶给「旧时间 - 新时间」这个偏移投票，偏移相同的相邻桶连成
       一段。段内偏移≈常数（内容没动），段与段之间偏移跳变 = 剪掉/插入了一段；
       开头片头那种重复内容会形成一段「旧时间往回跳」的独立段，所以字幕会在
       片头位置再出现一次（--no-repeat 只留正文那一次）；
    4. 每段用最小二乘拟合「旧时间 = a × 新时间 + b」（a 吸在 1 附近，兼容
       重编码带来的轻微变速），再把每条旧字幕按落在哪一段映射到新时间：
       整段被剪掉的直接丢弃，被剪点劈开的只留留下的那截；
    5. 目录：读同目录的 *.plan.txt（标题原样照搬），把每段的旧区间按同样的
       映射换成新时间，重排重编号后写成 `<计划名去掉 .plan>info.txt`。

特点:
    - 只改时间轴，字幕文本一个字都不改（（主播）这类前缀原样保留）
    - 对齐本身是纯 CPU 的文本处理（几秒钟），只有第一步识别要跑模型
    - 新旧文本有明显差异（纠错过的旧字幕 vs 原始 ASR）也能对上，靠的是多数投票
    - 统计和告警都打到 stderr；--report 会输出每段的偏移、残差、没对上的区间

用法:
    python rebuild_caption.py "video_source\\xxx_edited.mp4"
    python rebuild_caption.py "xxx_edited.mp4" --srt "原片.srt"
    python rebuild_caption.py "xxx_edited.mp4" --srt video_source\\分片文件夹
    python rebuild_caption.py "xxx_edited.mp4" --asr 已有.asr.jsonl   # 不重新识别
    python rebuild_caption.py "xxx_edited.mp4" -o out.srt --no-repeat
    python rebuild_caption.py "xxx_edited.mp4" --plan "xxx.plan.txt" --toc-out "目录.txt"
    python rebuild_caption.py "xxx_edited.mp4" --report              # 附带对齐报告
    python rebuild_caption.py "xxx_edited.mp4" --dry-run               # 只算不写
"""

import os
import re
import sys
import json
import math
import time
import bisect
import argparse
import subprocess

import numpy as np

from auto_edit import fmt_hms, probe_duration
from edit_with_caption import (
    read_srt, write_srt_file, log, list_files, build_parts, match_aux_files,
    VIDEO_EXTS, SRT_EXTS,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 对齐参数
#   锚点 = 新旧文本共有的 NGRAM 个连续汉字。8 个字基本唯一，又能容忍 1~2 个
#   ASR 错字；投票按 BIN_SEC 分桶，桶内偏移相差不超过 OFFSET_TOL 的算同意。
# ---------------------------------------------------------------------------
NGRAM = 8                   # 锚点 n-gram 长度（字符）
BIN_SEC = 2.0               # 偏移投票的时间桶宽度（秒）
OFFSET_TOL = 1.5            # 同一个桶里算「同一个偏移」的容差（秒）
MIN_BIN_VOTES = 2           # 一个桶至少几条锚点同意才采信
MAX_BIN_GAP = 3             # 连续几个空桶仍当作同一段（ASR 漏字 / 短暂静音）
MIN_RUN_ANCHORS = 3         # 一段至少几条锚点
MIN_RUN_SPAN = 1.0          # 一段至少覆盖多少秒新时间（片头蒙太奇很多短镜头）
RUN_MIN_DENSITY = 0.35      # 一段里平均每个桶至少几条锚点（太稀的当噪声）
RUN_MAX_RESID = 2.0         # 拟合后锚点残差的中位数上限（秒），超过就当噪声段
MERGE_GAP_TOL = 2.0         # 两段之间的新旧空档差不到这么多秒 = 内容连续，合并
INTERP_MAX_GAP = 3.0        # 段与段之间这么短的旧时间空档，允许线性插值补字幕
SLOPE_RANGE = (0.97, 1.03)  # 段内斜率范围（内容没动，只允许重编码级别的偏差）
REFINE_ROUNDS = 2           # 剔除错锚点后再投票的轮数（重复台词会拉偏偏移）
REFINE_TOL = 3.0            # 和上一遍映射差超过这么多秒的锚点，当「重复内容」剔掉
REFINE_RADIUS = 30.0        # 附近这么多秒内没有已知段时，不剔（那一块还没对出来）
ISLAND_TOL = 2.0            # 前后两段偏移一致、夹在中间的段却不同 -> 中间那段是噪声
RUN_EDGE_PAD = 0.8          # 段的旧时间两端各往外放一点（锚点是 8 字词，两头天然short一截）

# ---------------------------------------------------------------------------
# 输出参数
# ---------------------------------------------------------------------------
CUE_MIN_SEC = 0.30          # 映射后短于这个的字幕（被剪点戈到的半句）丢掉
CUE_JOIN_GAP = 0.25         # 同一条字幕被段边界切开、但间隔小于这个就并回一条
CACHE_DIR_NAME = "rebuild_cache"    # 新视频 ASR 缓存目录（放在视频旁边）
SPEAKER_FMT = "（{name}）"   # 旧字幕里的（主播）前缀，输出时原样还原


# ---------------------------------------------------------------------------
# 文本归一化 / 字符时间轴
# ---------------------------------------------------------------------------
def norm_chars(text):
    """只留汉字 / 字母 / 数字（标点、空白、表情、英文空格全部丢掉）。"""
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def build_track(items):
    """字幕 / ASR 条目 -> (字符序列, 每个字符的大致时间)。

    字符时间在它所属条目的 [start, end] 里按字数均摊——中文 ASR 常常一句
    一长条，均摊比用整条起点准得多（否则整句的字都挤在起点上）。
    """
    chars, times = [], []
    for it in items:
        start, end = float(it["start"]), float(it["end"])
        text = norm_chars(it.get("text"))
        if not text:
            continue
        span = max(end - start, 0.01)
        n = len(text)
        for j, ch in enumerate(text):
            chars.append(ch)
            times.append(start + span * (j + 0.5) / n)
    return "".join(chars), times


def collect_anchors(old_chars, old_times, new_chars, new_times):
    """新文本每个 n-gram 在旧文本里的所有出现位置 -> [(新时间, 旧时间)]。"""
    if len(old_chars) < NGRAM or len(new_chars) < NGRAM:
        return []
    index = {}
    for i in range(len(old_chars) - NGRAM + 1):
        gram = old_chars[i:i + NGRAM]
        lst = index.get(gram)
        if lst is None:
            index[gram] = [i]
        else:
            lst.append(i)

    anchors = []
    for j in range(len(new_chars) - NGRAM + 1):
        pos = index.get(new_chars[j:j + NGRAM])
        if not pos:
            continue
        nt = new_times[j]
        for i in pos:
            anchors.append((nt, old_times[i]))
    return anchors


# ---------------------------------------------------------------------------
# 偏移投票 -> 对齐段
#   「旧时间 - 新时间」在一段没被动过的内容里是常数（重编码带来的轻微变速
#   由后面的线性拟合吸收），剪掉一段之后这个常数会跳变，所以按时间桶投票
#   就能把新视频切成若干「偏移恒定」的段。
# ---------------------------------------------------------------------------
def vote_offsets(anchors):
    """按时间桶投票，返回 {桶序号: (偏移, 同意的锚点列表)}。"""
    buckets = {}
    for nt, ot in anchors:
        buckets.setdefault(int(nt // BIN_SEC), []).append((nt, ot))

    votes = {}
    for b, pairs in buckets.items():
        offsets = [ot - nt for nt, ot in pairs]
        # 滑窗找「同意者最多」的偏移（不量化，避免 3.24 和 3.26 被拆到两边）
        best_off, best_cnt = None, -1
        for off in offsets:
            cnt = sum(1 for d in offsets if abs(d - off) <= OFFSET_TOL)
            if cnt > best_cnt:
                best_off, best_cnt = off, cnt
        agree = [p for p in pairs if abs((p[1] - p[0]) - best_off) <= OFFSET_TOL]
        if len(agree) < MIN_BIN_VOTES:
            continue
        agree.sort(key=lambda p: p[1] - p[0])
        votes[b] = (agree[len(agree) // 2][1] - agree[len(agree) // 2][0], agree)
    return votes


def fit_affine(pairs):
    """(新时间, 旧时间) -> (a, b, 残差中位数)，旧时间 ≈ a × 新时间 + b。"""
    nt = np.array([p[0] for p in pairs], dtype=float)
    ot = np.array([p[1] for p in pairs], dtype=float)
    a, b = 1.0, float(np.median(ot - nt))
    if len(pairs) >= 2 and np.ptp(nt) > 0.5:
        try:
            slope, inter = np.polyfit(nt, ot, 1)
            # 段内内容没动，斜率只允许被重编码推歪一点点；离谱就当噪声
            if SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1] and abs(slope - 1.0) >= 0.002:
                a, b = float(slope), float(inter)
        except (np.linalg.LinAlgError, ValueError):
            pass
    resid = float(np.median(np.abs(ot - (a * nt + b))))
    return a, b, resid


def make_run(pairs):
    a, b, resid = fit_affine(pairs)
    nt = [p[0] for p in pairs]
    new_lo, new_hi = min(nt), max(nt)
    return {"a": a, "b": b, "resid": resid, "pairs": list(pairs),
            "new_lo": new_lo, "new_hi": new_hi,
            "old_lo": a * new_lo + b, "old_hi": a * new_hi + b}


def run_ok(run):
    """这一段的质量够不够（锚点数、覆盖时长、拟合残差、锚点密度）。"""
    span = run["new_hi"] - run["new_lo"]
    nbins = max(1.0, span / BIN_SEC + 1.0)
    return (len(run["pairs"]) >= MIN_RUN_ANCHORS
            and span >= MIN_RUN_SPAN
            and run["resid"] <= RUN_MAX_RESID
            and len(run["pairs"]) / nbins >= RUN_MIN_DENSITY)


def build_runs(votes):
    """把偏移相同的相邻桶连成段（先不筛，短碎段留给 merge_runs 消化）。"""
    groups, cur = [], None
    for b in sorted(votes):
        off, pairs = votes[b]
        if cur is None:
            cur = {"bins": [b], "pairs": list(pairs), "offs": [off]}
            continue
        med = float(np.median(cur["offs"]))
        if b - cur["bins"][-1] <= MAX_BIN_GAP and abs(off - med) <= OFFSET_TOL:
            cur["bins"].append(b)
            cur["offs"].append(off)
            cur["pairs"].extend(pairs)
        else:
            groups.append(cur)
            cur = {"bins": [b], "pairs": list(pairs), "offs": [off]}
    if cur:
        groups.append(cur)

    return [make_run(g["pairs"]) for g in groups if len(g["pairs"]) >= 2]


def merge_runs(runs):
    """把「中间只是没说话 / ASR 漏字」的相邻段并起来（偏移对得上才算连续）。

    碎段先并进邻居再筛质量：这样「内容其实连续、只是锚点稀」的地方不会因为
    段太短被整块丢掉（那会让那一带的字幕全部消失），而真正的噪声段孤立着，
    合并后仍然过不了 run_ok()，照样被筛掉。
    """
    merged = []
    for r in sorted(runs, key=lambda x: x["new_lo"]):
        if merged:
            p = merged[-1]
            new_gap = r["new_lo"] - p["new_hi"]
            old_gap = r["old_lo"] - p["old_hi"]
            if (-0.5 <= new_gap <= max(0.0, old_gap) + MERGE_GAP_TOL
                    and abs(old_gap - new_gap) <= MERGE_GAP_TOL
                    and abs(p["a"] - r["a"]) <= 0.005):
                merged[-1] = make_run(p["pairs"] + r["pairs"])
                continue
        merged.append(dict(r))
    return [r for r in merged if run_ok(r)]


def nearest_run(runs, new_t):
    """离某个新时间最近的段 -> (段, 距离秒)。"""
    best, best_d = None, float("inf")
    for r in runs:
        if r["new_lo"] <= new_t <= r["new_hi"]:
            return r, 0.0
        d = min(abs(new_t - r["new_lo"]), abs(new_t - r["new_hi"]))
        if d < best_d:
            best, best_d = r, d
    return best, best_d


def prune_anchors(anchors, runs, tol=REFINE_TOL, radius=REFINE_RADIUS):
    """剔掉和当前映射对不上的锚点——多半是「同一句话在别处也出现」引来的。

    只在附近（radius 内）有已知段时才剔：那一块还没对出来的地方不能瞎判。
    """
    kept = []
    for new_t, old_t in anchors:
        run, dist = nearest_run(runs, new_t)
        if run is None or dist > radius:
            kept.append((new_t, old_t))
            continue
        if abs(run["a"] * new_t + run["b"] - old_t) <= tol:
            kept.append((new_t, old_t))
    return kept


def drop_islands(runs, tol=ISLAND_TOL):
    """丢掉夹在两段中间、但偏移明显不对的短段。

    典型来源：同一句话（口头禅 / 重复台词）在别处也出现过，几秒钟的锚点全投到
    了那个位置，就在两段正确映射中间冒出一个「孤岛」。前后两段偏移一致说明
    中间根本没有剪辑点，所以夹在中间那个不同偏移的段一定是噪声。
    """
    if len(runs) < 3:
        return runs
    order = sorted(runs, key=lambda r: r["new_lo"])
    out = []
    for i, r in enumerate(order):
        if 0 < i < len(order) - 1:
            prev, nxt = order[i - 1], order[i + 1]
            if (abs(prev["b"] - nxt["b"]) <= tol
                    and abs(r["b"] - prev["b"]) > tol):
                continue                      # 孤岛，丢掉
        out.append(r)
    return out


def pad_runs(runs, pad=RUN_EDGE_PAD):
    """每段旧时间两端各往外放一点点，把被剪点戈在段边缘的句子找回来。

    旧时间上相邻的两段之间只留它们本来的空档，不会互相叠（叠了会让同一句
    话被映射两次）。
    """
    if pad <= 0 or not runs:
        return runs
    out = [dict(r) for r in runs]
    order = sorted(range(len(out)), key=lambda i: out[i]["old_lo"])
    for k in range(len(order) - 1):
        a, b = out[order[k]], out[order[k + 1]]
        if b["old_lo"] >= a["old_hi"]:        # 旧时间递增（中间是剪掉的一段）
            a["old_hi"] = min(a["old_hi"] + pad, b["old_lo"])
            b["old_lo"] = max(b["old_lo"] - pad, a["old_hi"])
    for r in out:
        r["new_lo"] = (r["old_lo"] - r["b"]) / r["a"]
        r["new_hi"] = (r["old_hi"] - r["b"]) / r["a"]
    return out


def align(anchors, rounds=REFINE_ROUNDS, verbose=False):
    """锚点 -> 对齐段：投票分段，剔噪点后再投票，最后修边。"""
    if not anchors:
        return [], anchors
    runs = merge_runs(build_runs(vote_offsets(anchors)))
    for _ in range(max(0, rounds)):
        if not runs:
            break
        clean = drop_islands(runs)
        kept = prune_anchors(anchors, clean)
        if len(clean) == len(runs) and len(kept) == len(anchors):
            break
        if verbose:
            log(f"      剔除 {len(anchors) - len(kept)} 个「重复内容」锚点后重算")
        anchors = kept
        runs = merge_runs(build_runs(vote_offsets(kept))) if kept else []
    return pad_runs(runs), anchors


def pick_main_chain(runs):
    """挑「旧时间递增」的最长链当正文：片头是重复内容，旧时间会往回跳。"""
    order = sorted(runs, key=lambda r: (r["new_lo"], r["old_lo"]))
    n = len(order)
    if n == 0:
        return []
    weight = [r["new_hi"] - r["new_lo"] for r in order]
    best, prev = list(weight), [-1] * n
    for i in range(n):
        for j in range(i):
            if order[j]["old_hi"] <= order[i]["old_lo"] + 1.0:
                cand = best[j] + weight[i]
                if cand > best[i]:
                    best[i], prev[i] = cand, j
    i = int(np.argmax(best))
    chain = []
    while i >= 0:
        chain.append(order[i])
        i = prev[i]
    return sorted(chain, key=lambda r: r["new_lo"])


# ---------------------------------------------------------------------------
# 旧时间 -> 新时间
# ---------------------------------------------------------------------------
def map_span(runs, start, end, min_sec=CUE_MIN_SEC):
    """把旧区间映射成新时间上的若干片（被剪点劈开时会有多片）。"""
    pieces = []
    for r in runs:
        lo = max(float(start), r["old_lo"])
        hi = min(float(end), r["old_hi"])
        if hi - lo < min_sec * r["a"]:
            continue
        pieces.append(((lo - r["b"]) / r["a"], (hi - r["b"]) / r["a"]))
    return pieces


def interp_span(runs, start, end, max_gap=INTERP_MAX_GAP):
    """落在两段之间的小空档（ASR 漏了几个字）-> 线性插值补回来。"""
    prev = nxt = None
    for r in runs:
        if r["old_hi"] <= start and (prev is None or r["old_hi"] > prev["old_hi"]):
            prev = r
        if r["old_lo"] >= end and (nxt is None or r["old_lo"] < nxt["old_lo"]):
            nxt = r
    if prev is None or nxt is None:
        return []
    gap_old = nxt["old_lo"] - prev["old_hi"]
    if gap_old <= 0 or gap_old > max_gap:
        return []
    # 只有「新时间上的间隔和旧时间上的间隔对得上」才说明内容真的连续，
    # 否则中间其实剪掉了一段，不能补
    if abs((nxt["new_lo"] - prev["new_hi"]) - gap_old) > 1.0:
        return []
    frac = (float(start) - prev["old_hi"]) / gap_old
    new_start = prev["new_hi"] + frac * (nxt["new_lo"] - prev["new_hi"])
    return [(new_start, new_start + (float(end) - float(start)))]


def join_pieces(pieces, gap=CUE_JOIN_GAP):
    """同一句被切开、但间隔很小的片并回一条。"""
    pieces = sorted(pieces)
    out = []
    for s, e in pieces:
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def rebuild_cues(cues, runs, keep_speaker=True,
                 min_sec=CUE_MIN_SEC, allow_interp=True):
    """旧字幕 -> 新时间轴字幕。返回 (字幕列表, 统计)。"""
    stats = {"dropped": 0, "split": 0, "extra": 0, "interp": 0}
    placements = []
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        pieces = join_pieces(map_span(runs, start, end, min_sec))
        if not pieces and allow_interp:
            pieces = interp_span(runs, start, end)
            if pieces:
                stats["interp"] += 1
        if not pieces:
            stats["dropped"] += 1
            continue
        if len(pieces) > 1:
            # 同一句被剪点劈开，或（片头）重复内容让它在别处又出现一次
            stats["split"] += 1
            stats["extra"] += len(pieces) - 1

        text = (cue.get("text") or "").strip()
        if not text:
            stats["dropped"] += 1
            continue
        speaker = (cue.get("speaker") or "").strip()
        if speaker and keep_speaker:
            text = SPEAKER_FMT.format(name=speaker) + text
        for s, e in pieces:
            placements.append({"start": s, "end": e, "text": text})
    return placements, stats


def finalize(placements, duration, min_sec=CUE_MIN_SEC):
    """排序、裁到视频范围内、去重叠、丢掉太短的。"""
    placements.sort(key=lambda x: (x["start"], x["end"]))
    out = []
    for item in placements:
        start = max(0.0, float(item["start"]))
        end = float(item["end"])
        if duration and duration > 0:
            end = min(end, duration)
        if out and start < out[-1]["end"] - 0.05:
            start = out[-1]["end"]          # 别压住上一条
        if end - start < min_sec:
            continue
        out.append({"start": start, "end": end, "text": item["text"]})
    return out


# ---------------------------------------------------------------------------
# 新视频的 ASR（默认复用 sensevoice_srt.py，结果缓存）
# ---------------------------------------------------------------------------
def read_asr_jsonl(path):
    """读 sensevoice_srt.py 的断点 JSONL（跳过 done 标记行）。"""
    items = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("done") or "text" not in rec or "start" not in rec:
                continue
            try:
                start, end = float(rec["start"]), float(rec["end"])
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            items.append({"start": start, "end": end,
                          "text": rec.get("text", "")})
    items.sort(key=lambda r: (r["start"], r["end"]))
    return items


def asr_cache_paths(video):
    folder = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    cache = os.path.join(folder, CACHE_DIR_NAME)
    return (os.path.join(cache, stem + ".srt"),
            os.path.join(cache, stem + ".asr.jsonl"))


def ensure_new_asr(video, args):
    """返回 (ASR 条目列表, 来源说明)。"""
    if args.asr:
        if not os.path.isfile(args.asr):
            raise SystemExit(f"找不到 --asr 文件：{args.asr}")
        if args.asr.lower().endswith(".jsonl"):
            return read_asr_jsonl(args.asr), args.asr
        return read_srt(args.asr), args.asr

    srt_path, jsonl_path = asr_cache_paths(video)
    if not args.force_asr and os.path.isfile(srt_path) and os.path.getsize(srt_path) > 0:
        log(f"复用已有识别结果: {srt_path}")
        return read_srt(srt_path), srt_path

    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "sensevoice_srt.py"), video,
           "-o", srt_path, "--jsonl", jsonl_path, "--device", args.device,
           "--no-streamer-mark"]
    if not args.correct:
        cmd.append("--no-correct")
    if os.path.isfile(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        cmd.append("--resume")              # 上次中断就接着跑

    log(f"识别新视频（第一次会慢一些，结果缓存在 {os.path.dirname(srt_path)}）...")
    t0 = time.time()
    code = subprocess.run(cmd, cwd=HERE).returncode
    if code != 0 or not os.path.isfile(srt_path):
        raise SystemExit(f"识别失败（退出码 {code}）：{video}")
    log(f"识别完成，耗时 {time.time() - t0:.0f}s")
    return read_srt(srt_path), srt_path


# ---------------------------------------------------------------------------
# 旧字幕
# ---------------------------------------------------------------------------
def _exclude_stem(path, stems):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.lower() in stems


def load_source_entries(video, srt_arg, src_video_dir=None, out_path=""):
    """读出「原始视频时间轴」上的字幕，返回 (条目列表, 说明)。"""
    stems = {os.path.splitext(os.path.basename(video))[0].lower()}
    if out_path:
        stems.add(os.path.splitext(os.path.basename(out_path))[0].lower())

    if srt_arg and os.path.isfile(srt_arg):
        return read_srt(srt_arg), srt_arg

    if srt_arg and os.path.isdir(srt_arg):
        vdir = src_video_dir or srt_arg
        videos = [p for p in list_files(vdir, VIDEO_EXTS)
                  if not _exclude_stem(p, stems)]      # 别把成片自己当分片
        if not videos:
            raise SystemExit(
                f"分片字幕模式要用源视频算各分片时长，但 {vdir} 里没有视频。\n"
                f"（可以把 --srt 指向单个覆盖全片的 .srt，或用 --src-video-dir 指定源视频目录）")
        files = [p for p in list_files(srt_arg, SRT_EXTS)
                 if not _exclude_stem(p, stems)]
        if not files:
            raise SystemExit(f"文件夹里没有可用的 .srt：{srt_arg}")
        if len(videos) != len(files):
            log(f"      ! 分片视频 {len(videos)} 个、字幕 {len(files)} 份，数量不一致；"
                f"分片时长算错会让后面的分片整体偏移，必要时用 --src-video-dir 指定源视频目录")
        parts = build_parts(videos)
        mapping = match_aux_files(parts, files, "字幕")
        entries = []
        for i, part in enumerate(parts):
            path = mapping.get(i)
            if not path:
                continue
            for e in read_srt(path):
                item = dict(e)
                item["start"] += part["offset"]
                item["end"] += part["offset"]
                entries.append(item)
        entries.sort(key=lambda e: (e["start"], e["end"]))
        desc = f"{srt_arg}（{len(mapping)}/{len(parts)} 个分片）"
        return entries, desc

    # 没给 --srt：优先用与视频同名的那份，其次用文件夹里唯一的那份
    folder = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    same = os.path.join(folder, stem + ".srt")
    if os.path.isfile(same):
        log(f"      用与视频同名的 {os.path.basename(same)} 当旧字幕"
            f"（如果它不是原片字幕，请用 --srt 指定）")
        return read_srt(same), same
    candidates = [p for p in list_files(folder, SRT_EXTS)
                  if not _exclude_stem(p, stems)]
    if len(candidates) == 1:
        log(f"      用文件夹里唯一的 {os.path.basename(candidates[0])} 当旧字幕")
        return read_srt(candidates[0]), candidates[0]
    if not candidates:
        raise SystemExit(f"找不到旧字幕，请用 --srt 指定（视频目录：{folder}）")
    raise SystemExit(
        "视频所在文件夹里有多个 .srt，不知道用哪个当旧字幕，请用 --srt 指定：\n  "
        + "\n  ".join(os.path.basename(p) for p in candidates))


# ---------------------------------------------------------------------------
# 剪辑计划（目录里的标题来源）
# ---------------------------------------------------------------------------
_PLAN_HEAD_RE = re.compile(
    r"^\s*\d+\s+(\d{1,2}:\d{2}:\d{2})\s*~\s*(\d{1,2}:\d{2}:\d{2})")
_PLAN_TITLE_RE = re.compile(r"^\s*标题\s*[:：]\s*(.+?)\s*$")


def parse_hms(text):
    h, m, s = (int(x) for x in text.split(":"))
    return h * 3600 + m * 60 + s


def read_plan_txt(path):
    """从 *.plan.txt 里抓「序号 / 旧时间区间 / 标题」，标题原样照搬。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    rows, cur = [], None
    for line in raw.split("\n"):
        m = _PLAN_HEAD_RE.match(line)
        if m:
            cur = {"start": parse_hms(m.group(1)), "end": parse_hms(m.group(2)),
                   "title": ""}
            rows.append(cur)
            continue
        m = _PLAN_TITLE_RE.match(line)
        if m and cur is not None and not cur["title"]:
            cur["title"] = m.group(1)
    return [r for r in rows if r["title"]]


def read_plan_json(path):
    """直接读 *.plan.json（逐片格式，按 title / id 归组成段）。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("segments") or []
    groups = {}
    for item in data:
        if not isinstance(item, dict) or "start" not in item:
            continue
        key = (item.get("id"), item.get("title"))
        g = groups.setdefault(key, {"start": float(item["start"]),
                                    "end": float(item["end"]),
                                    "title": item.get("title") or ""})
        g["start"] = min(g["start"], float(item["start"]))
        g["end"] = max(g["end"], float(item["end"]))
    return [g for g in groups.values() if g["title"]]


def _name_key(name):
    """文件名比相似度用：去掉 _、空格、大小写差异。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", name.lower())


def find_plan(video, plan_arg):
    """找剪辑计划：显式指定 > 视频同目录里唯一的那份 > 名字最像的那份。"""
    if plan_arg:
        if not os.path.isfile(plan_arg):
            raise SystemExit(f"找不到剪辑计划：{plan_arg}")
        return plan_arg
    folder = os.path.dirname(os.path.abspath(video))
    if not os.path.isdir(folder):
        return None
    plans = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if n.lower().endswith(".plan.txt") and
             os.path.isfile(os.path.join(folder, n))]
    if not plans:
        return None
    if len(plans) == 1:
        return plans[0]
    stem = _name_key(os.path.splitext(os.path.basename(video))[0])
    for p in plans:
        name = _name_key(re.sub(r"(?i)\.plan\.txt$", "", os.path.basename(p)))
        if name and (name == stem or stem.startswith(name) or name.startswith(stem)):
            return p
    log(f"[警告] {folder} 里有 {len(plans)} 份计划，不知道用哪份，跳过目录生成"
        f"（可用 --plan 指定）")
    return None


def default_toc_path(plan_path):
    """<计划名去掉 .plan.txt>info.txt —— 与 奶茶星铁4.7info.txt 的命名一致。"""
    folder = os.path.dirname(os.path.abspath(plan_path))
    base = os.path.basename(plan_path)
    base = re.sub(r"(?i)\.plan\.txt$", "", base)
    if base.lower().endswith(".txt"):
        base = base[:-4]
    return os.path.join(folder, base + "info.txt")


def build_toc(rows, runs, duration):
    """计划里的旧区间 -> 新时间，返回 [(新起点, 新终点, 标题)]。"""
    out, skipped = [], 0
    for row in rows:
        pieces = map_span(runs, row["start"], row["end"])
        if not pieces:
            skipped += 1
            continue
        # 片头那种重复内容会出现两片，目录按正文那一片（更长、更靠后）算
        pieces.sort(key=lambda p: (p[1] - p[0], p[0]))
        start, end = pieces[-1]
        if duration and duration > 0:
            start, end = max(0.0, start), min(float(duration), end)
        if end - start < 1.0:
            skipped += 1
            continue
        out.append((start, end, row["title"]))
    out.sort(key=lambda x: x[0])
    return out, skipped


def write_toc(rows, path):
    """写成 奶茶星铁4.7info.txt 那种目录格式：  [i/N] HH:MM:SS ~ HH:MM:SS 标题。"""
    lines = []
    for i, (start, end, title) in enumerate(rows, 1):
        lines.append(f"  [{i}/{len(rows)}] "
                     f"{fmt_hms(math.floor(start))} ~ "
                     f"{fmt_hms(math.floor(end))} {title}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(rows)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(path, info):
    lines = ["rebuild_caption 对齐报告", ""]
    for key, value in info.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend("  " + str(v) for v in value)
        else:
            lines.append(f"{key}: {value}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def merge_intervals(pairs):
    out = []
    for start, end in sorted(pairs):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def uncovered_spans(new_items, subs, min_span=3.0):
    """新视频里「有语音但没字幕」的区间——可能是新内容，也可能对齐失败。"""
    if not subs:
        return [(float(i["start"]), float(i["end"])) for i in new_items
                if float(i["end"]) - float(i["start"]) >= min_span]
    merged = merge_intervals([(s["start"], s["end"]) for s in subs])
    starts = [a for a, _ in merged]
    holes = []
    for item in new_items:
        start, end = float(item["start"]), float(item["end"])
        if end - start < 1.0:
            continue
        i = bisect.bisect_right(starts, start) - 1
        covered = ((i >= 0 and merged[i][1] > start)
                   or (i + 1 < len(merged) and merged[i + 1][0] < end))
        if covered:
            continue
        if holes and start - holes[-1][1] < 2.0:
            holes[-1] = (holes[-1][0], max(holes[-1][1], end))
        else:
            holes.append((start, end))
    return [h for h in holes if h[1] - h[0] >= min_span]


def report_text(entries, new_items, runs, duration, stats, cues_in, subs_out,
                max_runs=12):
    """把对齐结果整理成人能看懂的一段文本（同时打到 stderr）。"""
    body = []
    body.append(f"旧字幕 {len(entries)} 条，新视频 ASR {len(new_items)} 条，"
                f"新视频时长 {fmt_hms(duration or 0)}")

    ordered = sorted(runs, key=lambda x: x["new_lo"])
    body.append(f"对齐段 {len(ordered)} 个")
    shown = ordered if len(ordered) <= max_runs else ordered[:max_runs - 3]
    for i, r in enumerate(shown, 1):
        body.append(
            f"  {i:02d}  新 {fmt_hms(r['new_lo'])} ~ {fmt_hms(r['new_hi'])}  "
            f"<- 旧 {fmt_hms(r['old_lo'])} ~ {fmt_hms(r['old_hi'])}  "
            f"a={r['a']:.4f} b={r['b']:+.1f}s  "
            f"锚点 {len(r['pairs'])}  残差 {r['resid']:.2f}s")
    if len(ordered) > len(shown):
        body.append(f"  ...（完整列表见 --report 写出的报告）")
        for i, r in enumerate(ordered[-3:], len(ordered) - 2):
            body.append(
                f"  {i:02d}  新 {fmt_hms(r['new_lo'])} ~ {fmt_hms(r['new_hi'])}  "
                f"<- 旧 {fmt_hms(r['old_lo'])} ~ {fmt_hms(r['old_hi'])}  "
                f"a={r['a']:.4f} b={r['b']:+.1f}s  "
                f"锚点 {len(r['pairs'])}  残差 {r['resid']:.2f}s")

    holes = uncovered_spans(new_items, subs_out)
    if holes:
        body.append(f"没对上的新区间 {len(holes)} 个（有语音但没字幕，"
                    f"可能是新内容或对齐失败）：")
        for start, end in holes[:10]:
            body.append(f"  {fmt_hms(start)} ~ {fmt_hms(end)}  ({end - start:.1f}s)")
        if len(holes) > 10:
            body.append(f"  ...（另外 {len(holes) - 10} 个）")
    else:
        body.append("没对上的新区间：无")

    body.append(f"旧字幕 {cues_in} 条 -> 新字幕 {len(subs_out)} 条"
                f"（整条被剪掉 {stats['dropped']}，被剪点 / 重复内容分成多片 "
                f"{stats['split']}（多出 {stats['extra']} 片），"
                f"插值补回 {stats['interp']}）")
    return body


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="把旧字幕重新对到新视频上，并按剪辑计划重生成新时间点的目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("video", help="新视频（剪过的那条）")
    ap.add_argument("--srt",
                    help="旧字幕：单个 .srt（覆盖原始全片时间轴），或分片字幕文件夹")
    ap.add_argument("--src-video-dir",
                    help="--srt 给文件夹时，用这里的源视频算分片时长（默认同 --srt 目录）")
    ap.add_argument("--plan", help="剪辑计划 *.plan.txt（默认用视频同目录里的那份）")
    ap.add_argument("--asr",
                    help="新视频的 ASR（.asr.jsonl 或 .srt）；给了就不调用 sensevoice_srt.py")
    ap.add_argument("-o", "--output", help="输出字幕（默认 <视频名>.srt）")
    ap.add_argument("--toc-out", help="输出目录（默认 <计划名去掉 .plan>info.txt）")
    ap.add_argument("--no-toc", action="store_true", help="不生成目录")
    ap.add_argument("--no-repeat", action="store_true",
                    help="片头那种重复内容不再出字幕（只保留正文那一次）")
    ap.add_argument("--no-speaker", action="store_true",
                    help="去掉（主播）这类前缀")
    ap.add_argument("--device", default="auto", help="识别用的设备：auto / cuda:0 / cpu")
    ap.add_argument("--correct", action="store_true",
                    help="识别时顺带让大模型纠错（默认关：对齐不需要纠错，还慢）")
    ap.add_argument("--force-asr", action="store_true", help="忽略缓存，重新识别")
    ap.add_argument("--report", nargs="?", const=True, default=False,
                    help="输出对齐报告（默认 <输出字幕名>_align.txt）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写文件")
    args = ap.parse_args(argv)

    video = args.video
    if not os.path.isfile(video):
        raise SystemExit(f"找不到视频：{video}")

    # ---- 输出路径 ----
    folder = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    out_path = args.output
    if not out_path:
        out_path = os.path.join(folder, stem + ".srt")
        if os.path.isfile(out_path):
            # 别覆盖掉正在当「旧字幕」用的那份
            out_path = os.path.join(folder, stem + ".rebuilt.srt")
    out_path = os.path.abspath(out_path)

    log(f"[1/4] 读旧字幕 ...")
    entries, src_desc = load_source_entries(video, args.srt,
                                            args.src_video_dir, out_path)
    if not entries:
        raise SystemExit(f"旧字幕是空的：{src_desc}")
    log(f"      旧字幕 {len(entries)} 条：{src_desc}")

    duration = 0.0
    try:
        duration = probe_duration(video)
    except Exception:                        # noqa: BLE001  时长只用于裁剪和报告
        log("      ! 读不到新视频时长，跳过越界裁剪")

    log(f"[2/4] 拿新视频的语音时间轴 ...")
    new_items, asr_desc = ensure_new_asr(video, args)
    if not new_items:
        raise SystemExit(f"新视频没有识别到语音：{asr_desc}")
    log(f"      ASR {len(new_items)} 条：{asr_desc}")

    log(f"[3/4] 对齐 ...")
    t0 = time.time()
    old_chars, old_times = build_track(entries)
    new_chars, new_times = build_track(new_items)
    log(f"      文本：旧 {len(old_chars)} 字 / 新 {len(new_chars)} 字")

    anchors = collect_anchors(old_chars, old_times, new_chars, new_times)
    log(f"      锚点 {len(anchors)} 个")
    runs, anchors = align(anchors, verbose=True)
    if not runs:
        raise SystemExit(
            "对齐失败：新视频的语音和旧字幕几乎对不上。\n"
            "（确认 --srt 给的是这条视频的原片字幕；用 --report 可以看细节）")

    if args.no_repeat and len(runs) > 1:
        body = pick_main_chain(runs)
        if body:
            log(f"      只保留正文：{len(body)}/{len(runs)} 段")
            runs = body

    placements, stats = rebuild_cues(entries, runs,
                                    keep_speaker=not args.no_speaker)
    subs = finalize(placements, duration)
    log(f"      对齐段 {len(runs)} 个，耗时 {time.time() - t0:.1f}s")

    for line in report_text(entries, new_items, runs, duration, stats,
                            len(entries), subs):
        log("      " + line)

    # 新视频的语音大面积没对上字幕，多半是 --srt 指错了文件
    holes = uncovered_spans(new_items, subs)
    speech = sum(max(0.0, i["end"] - i["start"]) for i in new_items)
    hole_sec = sum(end - start for start, end in holes)
    if speech > 0 and hole_sec / speech > 0.4:
        log(f"[警告] 新视频有 {hole_sec:.0f}s 语音（{hole_sec / speech * 100:.0f}%）"
            f"没对到字幕上，确认 --srt 是不是这条视频的原片字幕")

    # ---- 目录 ----
    toc_rows, toc_path, skipped = [], None, 0
    plan_path = None if args.no_toc else find_plan(video, args.plan)
    if plan_path:
        try:
            plan_rows = (read_plan_json(plan_path)
                         if plan_path.lower().endswith(".json")
                         else read_plan_txt(plan_path))
        except (OSError, ValueError) as exc:
            log(f"[警告] 读剪辑计划失败，跳过目录：{exc}")
            plan_rows = []
        if not plan_rows:
            log(f"[警告] 剪辑计划里没找到「标题」，跳过目录：{plan_path}")
        else:
            toc_rows, skipped = build_toc(plan_rows, runs, duration)
            toc_path = args.toc_out or default_toc_path(plan_path)
            log(f"      目录 {len(toc_rows)} 条（剪掉了 {skipped} 条）："
                f"{os.path.basename(plan_path)}")
    elif not args.no_toc:
        log("[提示] 没找到剪辑计划（*.plan.txt），这次只重建字幕")

    # ---- 写文件 ----
    log(f"[4/4] 写文件 ...")
    if args.dry_run:
        for s in subs[:5]:
            log(f"      预览 {s['start']:8.2f}s --> {s['end']:8.2f}s  {s['text']}")
        log(f"      （--dry-run：共 {len(subs)} 条，未写文件）")
        return 0

    count = write_srt_file(subs, out_path)
    log(f"      字幕  {count} 条 -> {out_path}")
    if toc_rows:
        n = write_toc(toc_rows, toc_path)
        log(f"      目录  {n} 条 -> {toc_path}")

    if args.report:
        path = args.report if isinstance(args.report, str) else \
            os.path.splitext(out_path)[0] + "_align.txt"
        info = {
            "新视频": video,
            "新视频时长": fmt_hms(duration or 0),
            "旧字幕": src_desc,
            "新视频 ASR": asr_desc,
            "字幕条数": f"{len(entries)} -> {len(subs)}（丢弃 {stats['dropped']}，"
                        f"分成多片 {stats['split']}（多出 {stats['extra']} 片），"
                        f"插值 {stats['interp']}）",
            "目录": f"{len(toc_rows)} 条（剪掉 {skipped}）" if toc_rows else "未生成",
        }
        write_report(path, {**info, "明细": report_text(
            entries, new_items, runs, duration, stats, len(entries), subs)})
        log(f"      报告  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

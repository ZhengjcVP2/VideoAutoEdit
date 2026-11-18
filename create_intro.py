#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_intro.py — 做一个 6K 的 3x3 九宫格片头

成品：5760x3240（9 块 1920x1080 拼成 3 行 3 列），总时长默认 15 秒。
每一块都从源素材里取「一段连续画面」（默认就是整条 15 秒连续播放），
并且尽量让 9 块的画面互不相同：

    1. 源素材可以是多个视频文件夹（每个文件夹只探测一级目录里的视频，
       子目录不算；文件夹内的多分片按字母顺序拼成它自己的一条时间轴，
       规则与 edit_with_caption.py 完全一致），也可以直接给视频文件；
    2. 9 块会分给各个文件夹：每个文件夹至少 1 块，素材越长分得越多
       （但不会超过它的片长允许的块数）；
       每个文件夹再把自己的时间轴均分成若干区段，每个区段只取 1 块，
       这样 9 块的拍摄时间天然拉得很开（跨剧情段落）；
    3. 每个区段内再采样若干个候选起点，抽取中间那一帧算一个
       「粗粒度缩略图特征」（4x3 网格的平均色），
       用贪婪策略选出彼此差异最大的一组（第一个取最有个性的）；
    4. 每个片段都保证落在同一个分片内部，不会跨文件；
    5. xstack 拼成 9 宫格后重新编码输出。

镜头对齐 / 编码复用 auto_edit.py 里的 FFMPEG、时长探测；分片排序、
时间轴偏移复用 edit_with_caption.py 里的 list_files / build_parts，
本程序不重复实现这些规则。

用法:
    python create_intro.py                              # 用下面 SOURCE 常量（可多个文件夹）
    python create_intro.py "video_source\\a"            # 指定一个文件夹
    python create_intro.py "video_source\\a" "video_source\\b"   # 多个文件夹，各取一级目录视频
    python create_intro.py video.mp4 -o intro.mp4       # 单个视频
    python create_intro.py --duration 10                # 总时长 10 秒
    python create_intro.py --no-diversity               # 不做画面差异挑选（直接用区段中点）
    python create_intro.py --tile-size 480x270 --duration 6 -o preview.mp4   # 快速预览
"""

import os
import sys
import argparse
import subprocess

import numpy as np

from auto_edit import FFMPEG, fmt_hms, require_tools, run_ffmpeg
from edit_with_caption import (
    VIDEO_EXTS, list_files, build_parts, timeline_duration,
)

# ---------------------------------------------------------------------------
# 输入 / 输出
# ---------------------------------------------------------------------------
SOURCE = [r"video_source\文七绝区零day2",
          r"video_source\文七绝区零day3",
          r"video_source\文七绝区零day4"]   # 视频文件夹（可多个），也可给单个视频文件
OUTPUT_PATH = r"video_source\文七绝区零intro.mp4"                    # 空 = <源目录名>_intro.mp4 放在源目录的上一级

# ---------------------------------------------------------------------------
# 画面参数
# ---------------------------------------------------------------------------
DURATION = 35.0             # 成片总时长（秒），每块也是这么长的连续画面
GRID = 3                    # 3x3
TILE_W, TILE_H = 1920, 1080  # 单块分辨率（3x3 拼起来就是 5760x3240）

# ---------------------------------------------------------------------------
# 挑选参数（让 9 块画面尽量不同）
# ---------------------------------------------------------------------------
DIVERSITY = True            # True = 在区段内按画面特征挑差异最大的一组
CANDIDATES_PER_ZONE = 7     # 每个区段采样几个候选起点
SIG_SIZE = (16, 9)          # 特征提取用的缩略图尺寸
SIG_CELLS = (4, 3)          # 特征网格（列 x 行），每格取平均色
MIN_BRIGHTNESS = 20.0       # 候选帧平均亮度下限（0~255），避开黑屏 / 淡入淡出
MIN_CONTRAST = 8.0          # 候选帧对比度下限（网格间标准差），避开纯色画面

# ---------------------------------------------------------------------------
# 编码参数
# ---------------------------------------------------------------------------
FPS = 30
CRF = 18
PRESET = "medium"
AUDIO_MODE = "none"         # none = 无声；first = 用第 1 块的音轨


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def parse_size(text):
    """解析 '1920x1080' -> (1920, 1080)。"""
    try:
        w, h = text.lower().split("x")
        w, h = int(w), int(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"分辨率格式应为 WxH，例如 1920x1080：{text!r}")
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError("宽高必须是正偶数（yuv420p 要求）")
    return w, h


def collect_groups(sources):
    """源素材 -> [{"source", "name", "parts"}, ...]，一个文件夹（或文件）算一组。

    每个文件夹只探测一级目录（list_files 不递归），子目录里的视频不参与。
    """
    groups = []
    for src in sources:
        if os.path.isdir(src):
            videos = list_files(src, VIDEO_EXTS)
            if not videos:
                log(f"  ! 跳过（一级目录里没有视频）: {src}")
                continue
            name = os.path.basename(os.path.normpath(os.path.abspath(src)))
            parts = build_parts(videos)
        elif os.path.isfile(src):
            name = os.path.basename(src)
            parts = build_parts([src])
        else:
            raise SystemExit(f"找不到源素材: {src}")
        for p in parts:
            p["group"] = name
        groups.append({"source": src, "name": name, "parts": parts})
    if not groups:
        raise SystemExit("没有可用的源素材")
    return groups


def default_output(sources):
    """没给 -o 时按源素材名推导输出路径：<父目录>/<名字>_intro.mp4。"""
    root = os.path.abspath(sources[0])
    if os.path.isdir(root):
        name = os.path.basename(os.path.normpath(root))
    else:
        name = os.path.splitext(os.path.basename(root))[0]
    if len(sources) > 1:
        name = f"{name}_等{len(sources)}个"
    return os.path.join(os.path.dirname(root), name + "_intro.mp4")


def allocate_tiles(totals, n, seg_len):
    """把 n 块分给各文件夹：每组至少 1 块，剩下按时长比例分（长假素材多分）。"""
    counts = [0] * len(totals)
    caps = [max(1, int(t // seg_len)) for t in totals]   # 片长最多能拆出几块

    # 先保证每个文件夹都露面（素材长的先拿）
    for i in sorted(range(len(totals)), key=lambda i: -totals[i])[:n]:
        counts[i] = 1

    while sum(counts) < n:
        cand = [i for i in range(len(totals)) if counts[i] < caps[i]]
        if not cand:      # 都不够长了：退回允许区段重叠，尽量把块数分满
            cand = list(range(len(totals)))
        i = max(cand, key=lambda i: totals[i] / (counts[i] + 1))
        counts[i] += 1
    return counts


def plan_zones(groups, counts, seg_len):
    """按每个文件夹分到的块数，把它自己的时间轴均分成若干区段。

    返回 [(parts, lo, hi), ...]，每块对应一个区段。
    """
    zones = []
    for g, k in zip(groups, counts):
        if k <= 0:
            continue
        total = timeline_duration(g["parts"])
        for lo, hi in even_windows(total, k, seg_len):
            zones.append((g["parts"], lo, hi))
    return zones


# ---------------------------------------------------------------------------
# 在分片内取连续片段
# ---------------------------------------------------------------------------
def fit_in_part(parts, t, seg_len):
    """把全局起点 t 夹进某个分片，保证 [t, t+seg_len] 不跨分片。

    返回 (part, 全局起点, 局部起点)。
    """
    for p in parts:
        a = p["offset"]
        b = p["offset"] + p["duration"] - seg_len
        if b < a:          # 这个分片装不下这么长的片段
            continue
        if t <= b:
            g = max(a, min(t, b))
            return p, g, g - a

    # 没有任何分片装得下（片长比总时长还短）：退到最长的分片，尽量取满
    p = max(parts, key=lambda x: x["duration"])
    local = max(0.0, min(t - p["offset"], max(p["duration"] - seg_len, 0.0)))
    return p, p["offset"] + local, local


def zone_candidates(parts, lo, hi, seg_len, n):
    """在全局区间 [lo, hi] 内取 n 个候选起点（都保证落在单个分片里）。"""
    targets = [lo + (hi - lo) * (k + 0.5) / n for k in range(n)]
    out, seen = [], set()
    for t in targets:
        p, g, local = fit_in_part(parts, t, seg_len)
        key = (p["index"], round(local, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({"part": p, "start": g, "local": local})
    return out


# ---------------------------------------------------------------------------
# 画面特征 + 差异最大化挑选
# ---------------------------------------------------------------------------
def frame_signature(path, t, size=SIG_SIZE, cells=SIG_CELLS):
    """抽一帧算粗粒度特征（cells 网格的平均 RGB），失败返回 None。"""
    w, h = size
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(t, 0.0):.3f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except OSError:
        return None
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        return None
    arr = np.frombuffer(proc.stdout[:w * h * 3], dtype=np.uint8)
    img = arr.reshape(h, w, 3).astype(np.float32)
    cw, ch = cells
    if w % cw or h % ch:
        return img.reshape(-1).mean() * np.ones(cw * ch * 3, dtype=np.float32)
    pooled = img.reshape(ch, h // ch, cw, w // cw, 3).mean(axis=(1, 3))
    return pooled.reshape(-1)


def signature_distance(a, b):
    """两个特征的平均绝对差（0~255 尺度）。"""
    if a is None or b is None:
        return 255.0
    return float(np.mean(np.abs(a - b)))


def is_usable(sig):
    """排除黑屏 / 淡入淡出 / 纯色画面（这种块拼进片头很难看）。"""
    if sig is None:
        return False
    return float(sig.mean()) >= MIN_BRIGHTNESS and float(sig.std()) >= MIN_CONTRAST


def _pos_key(cand):
    """候选位置的唯一标识（同一个文件 + 同一个局部起点）。"""
    return (cand["part"]["path"], round(cand["local"], 2))


def pick_positions(zone_specs, seg_len, n_cand, diversity=True):
    """为每块挑一个片段：默认按区段取样 + 贪婪最大化彼此画面差异。

    zone_specs 是 [(parts, lo, hi), ...]（每块的区段），parts 是该块所属
    文件夹自己的时间轴。
    """
    zones = []
    for parts, lo, hi in zone_specs:
        cands = zone_candidates(parts, lo, hi, seg_len, n_cand)
        if not cands:
            p, g, local = fit_in_part(parts, lo, seg_len)
            cands = [{"part": p, "start": g, "local": local}]
        zones.append(cands)

    if diversity:
        log(f"抽取画面特征（{sum(len(z) for z in zones)} 帧）...")
        for cands in zones:
            for c in cands:
                # 取片段中间那一帧做代表
                c["sig"] = frame_signature(c["part"]["path"],
                                           c["local"] + seg_len / 2.0)
            # 尽量把黑屏 / 纯色画面剔掉；全都不可用时再退回原始候选
            usable = [c for c in cands if is_usable(c["sig"])]
            if usable:
                cands[:] = usable

    picks, used = [], set()
    for zi, cands in enumerate(zones):
        # 尽量不重复同一个位置（区段重叠时才可能撞车）
        fresh = [c for c in cands if _pos_key(c) not in used]
        pool = fresh or cands
        if not diversity:
            best = pool[len(pool) // 2]
        elif not picks:
            # 第一块取「最有个性」的：离本区段所有候选的平均特征最远
            sigs = [c["sig"] for c in pool if c["sig"] is not None]
            if sigs:
                mean_sig = np.mean(sigs, axis=0)
                best = max(pool, key=lambda c: signature_distance(c["sig"], mean_sig))
            else:
                best = pool[len(pool) // 2]
        else:
            # 后面的块：选跟已选画面"最不像"的那个
            best = max(pool, key=lambda c: min(
                signature_distance(c["sig"], p.get("sig")) for p in picks))
        best = dict(best)
        best["zone"] = zi
        used.add(_pos_key(best))
        picks.append(best)
    return picks


def even_windows(total, n, seg_len):
    """把 [0, total] 均分成 n 个区段；素材太短时允许区段重叠。"""
    width = total / n
    if width < seg_len:
        # 素材不足以让 n 段互不重叠：仍按区段中心铺开，允许重叠
        log(f"  ! 素材总长 {fmt_hms(total)} 较短，{n} 段可能出现重叠")
    return [(i * width, min(total, (i + 1) * width)) for i in range(n)]


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def build_filter(picks, tile_w, tile_h, grid, fps):
    """生成 xstack 的 filter_complex。"""
    parts = []
    for i in range(len(picks)):
        parts.append(
            f"[{i}:v]scale={tile_w}:{tile_h}"
            f":force_original_aspect_ratio=decrease,"
            f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p[v{i}]")
    layout = "|".join(
        f"{(i % grid) * tile_w}_{(i // grid) * tile_h}" for i in range(len(picks)))
    joined = "".join(f"[v{i}]" for i in range(len(picks)))
    parts.append(f"{joined}xstack=inputs={len(picks)}:layout={layout}"
                 f":fill=black[out]")
    return ";".join(parts)


def render(picks, out_path, duration, tile_w, tile_h, grid, fps, crf, preset,
           audio_mode):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in picks:
        # -ss 放在 -i 前面：快速定位；因为要重新编码，输出仍然对齐到该时间点
        cmd += ["-ss", f"{p['local']:.3f}", "-t", f"{duration:.3f}",
                "-i", p["part"]["path"]]

    cmd += ["-filter_complex", build_filter(picks, tile_w, tile_h, grid, fps),
            "-map", "[out]", "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-r", str(fps)]

    if audio_mode == "first":
        # 只保留第 1 块的音轨（9 条音轨混在一起没法听）
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    if out_path.lower().endswith((".mp4", ".mov")):
        cmd += ["-movflags", "+faststart"]
    cmd.append(out_path)

    log(f"渲染  {tile_w * grid}x{tile_h * grid}（{len(picks)} 块，"
        f"{duration:g}s，{fps}fps，crf {crf}，preset {preset}）...")
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="用 9 段连续画面拼一个 3x3 的 6K 片头（默认 15 秒）")
    parser.add_argument("sources", nargs="*", default=None,
                        help="一个或多个视频文件夹（只探测一级目录）或视频文件；"
                             "省略时用脚本里的 SOURCE 常量："
                             f"{'、'.join(SOURCE) if SOURCE else '（未设置）'}")
    parser.add_argument("-o", "--output", default=OUTPUT_PATH,
                        help="输出视频路径")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help=f"总时长秒数，默认 {DURATION:g}")
    parser.add_argument("--tile-size", type=parse_size,
                        default=(TILE_W, TILE_H),
                        help=f"单块分辨率，默认 {TILE_W}x{TILE_H}")
    parser.add_argument("--fps", type=int, default=FPS,
                        help=f"输出帧率，默认 {FPS}")
    parser.add_argument("--crf", type=int, default=CRF,
                        help=f"x264 质量（越小越好），默认 {CRF}")
    parser.add_argument("--preset", default=PRESET,
                        help=f"x264 preset，默认 {PRESET}")
    parser.add_argument("--audio", choices=["none", "first"], default=AUDIO_MODE,
                        help="音轨：none = 无声（默认），first = 用第 1 块的声音")
    parser.add_argument("--candidates", type=int, default=CANDIDATES_PER_ZONE,
                        help=f"每个区段的候选数，默认 {CANDIDATES_PER_ZONE}")
    parser.add_argument("--no-diversity", dest="diversity",
                        action="store_false", default=DIVERSITY,
                        help="不比较画面差异，直接用每个区段的中点")
    args = parser.parse_args()

    require_tools()

    if args.duration <= 0:
        raise SystemExit("总时长必须大于 0")
    if args.candidates < 1:
        raise SystemExit("候选数至少为 1")

    n = GRID * GRID
    sources = list(args.sources) or list(SOURCE)
    if not sources:
        parser.error("没有指定源素材，脚本里的 SOURCE 也是空的")
    groups = collect_groups(sources)

    # 短于一个片段的文件夹没法取连续画面，直接跳过
    usable = []
    for g in groups:
        t = timeline_duration(g["parts"])
        if t < args.duration:
            log(f"  ! 跳过 {g['name']}：总长 {fmt_hms(t)} 不足 {args.duration:g}s")
            continue
        usable.append(g)
    if not usable:
        raise SystemExit(f"所有源素材都短于一个 {args.duration:g}s 的片段")
    groups = usable

    if len(groups) > n:
        log(f"  ! 素材文件夹 {len(groups)} 个 > {n} 块，只按顺序用前 {n} 个")
        groups = groups[:n]

    out_path = args.output or default_output(sources)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    log(f"源素材   : {len(groups)} 个")
    for g in groups:
        log(f"           - {g['name']}  {len(g['parts'])} 分片，"
            f"总长 {fmt_hms(timeline_duration(g['parts']))}")
    log(f"成片     : {args.tile_size[0] * GRID}x{args.tile_size[1] * GRID}，"
        f"{args.duration:g}s，{args.fps}fps")
    log(f"挑选     : {'按画面差异' if args.diversity else '区段中点'}"
        f"（每块 {args.duration:g}s 连续画面）")

    counts = allocate_tiles([timeline_duration(g["parts"]) for g in groups],
                            n, args.duration)
    for g, k in zip(groups, counts):
        log(f"分配     : {g['name']} 取 {k} 块")

    zone_specs = plan_zones(groups, counts, args.duration)
    picks = pick_positions(zone_specs, args.duration, args.candidates,
                           args.diversity)

    log(f"\n{len(picks)} 块素材：")
    for i, p in enumerate(picks, 1):
        if args.diversity and len(picks) > 1:
            others = [q for q in picks if q is not p]
            diff = min(signature_distance(p.get("sig"), q.get("sig"))
                       for q in others)
            note = f"  画面差异 {diff:.0f}"
        else:
            note = ""
        log(f"  {i:02d} [{p['part'].get('group', '?')}] "
            f"{fmt_hms(p['start'])} ~ "
            f"{fmt_hms(p['start'] + args.duration)}  "
            f"{os.path.basename(p['part']['path'])} @ "
            f"{fmt_hms(p['local'])}{note}")

    render(picks, out_path, args.duration, args.tile_size[0], args.tile_size[1],
           GRID, args.fps, args.crf, args.preset, args.audio)
    log(f"完成: {out_path}")


if __name__ == "__main__":
    main()

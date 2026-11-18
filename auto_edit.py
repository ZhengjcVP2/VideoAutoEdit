#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_edit.py — 按 txt 里记录的秒区间剪辑视频并拼接

流程:
    视频 + 区间txt  --(ffmpeg -c copy 逐段切)-->  临时分片
                   --(concat demuxer 拼接)-->    成品视频

全程 **stream copy**，不重新编码：画质与源视频逐帧一致，单位时间码率不变，
速度只受磁盘 I/O 限制（8 小时素材通常 1~3 分钟跑完）。

区间 txt 格式（制表符 / 空格 / 逗号分隔均可，只取每行前两列，单位：秒）:
    754.000	796.000	42.000
    1906.000	1932.000	26.000
    ...

偏移规则:
    每段实际使用区间 = (start - offset1, end - offset2)

重复片段补偿:
    弹幕 txt 里的时间是真实时间，而直播回放可能每 30 分钟重复一小段
    （默认 4 秒）。每遇到一次重复，之后的视频时间就比真实时间滞后 4 秒。
    打开 SHIFT_FOR_DUPLICATES（或命令行加 --shift-duplicates）后，剪辑点按
        video_time = real_time + DUP_LEN * floor(real_time / DUP_INTERVAL)
    推后；DUP_LEN / DUP_INTERVAL 都是可改的变量。

剪辑点音频对齐:
    每个剪辑点都会在 ±SNAP_MAX_SHIFT 秒内寻找静音间隙（默认 5 秒）：
    开始点只往前移，结束点只往后移，切到最长静音段的中间；
    找不到静音时退回到音量最低的帧，让断句尽量完整。
    可用 --no-snap 关闭，或用 --snap-shift 修改最大移动秒数。

用法:
    python auto_edit.py                        # 默认 offset1=10s, offset2=5s
    python auto_edit.py -s 10 -e 5
    python auto_edit.py -s 0 -e 0              # 不做偏移，按 txt 原样切
    python auto_edit.py --shift-duplicates     # 按每 30 分钟 +4s 推后剪辑点
    python auto_edit.py --dup-len 4 --dup-interval 1800
    python auto_edit.py --snap-shift 5         # 边界最多移动 5 秒找静音
    python auto_edit.py --no-snap              # 关闭边界音频对齐
    python auto_edit.py --keep-temp            # 保留临时分片便于排查
"""

import os
import re
import shutil
import argparse
import tempfile
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# 输入 / 输出路径（暂时以变量形式写在程序内）
# ---------------------------------------------------------------------------
VIDEO_PATH = r"sample\testvideo_zzz_day1.mp4"                  # 待剪辑的视频
REGIONS_TXT = r"sample\test1_analyzed.txt"        # 区间列表（每行前两列是秒）
OUTPUT_PATH = r"sample\test1_edited.mp4"          # 成品路径

# 偏移默认值（秒）：开始时间减 offset1，结束时间减 offset2
DEFAULT_OFFSET1 = 30.0
DEFAULT_OFFSET2 = 15.0
# 重复片段补偿开关：打开后认为视频每 DUP_INTERVAL 秒会重复 DUP_LEN 秒
SHIFT_FOR_DUPLICATES = True
DUP_INTERVAL = 30 * 60          # 重复片段出现的间隔（秒）
DUP_LEN = 4.0                   # 每个重复片段的长度（秒），可按实际视频修改
# 剪辑点音频对齐：在边界附近找静音间隙，让断句完整
SNAP_BOUNDARIES = True
SNAP_MAX_SHIFT = 5.0            # 剪辑点最多移动的秒数（开始往前 / 结束往后）
SNAP_PAD = 0.5                  # 搜索窗口两端多看的上下文（秒）
SNAP_MIN_SILENCE = 0.08         # 至少这么长的静音才作为候选切点
SNAP_FRAME = 0.030              # RMS 帧长（秒）
SNAP_HOP = 0.010                # RMS 帧移（秒）
AUDIO_SR = 16000                # 分析音频时重采样到的采样率
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def fmt_hms(seconds):
    """把秒格式化成 HH:MM:SS。"""
    x = int(max(round(float(seconds)), 0))
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def require_tools():
    """确认 ffmpeg / ffprobe 可用。"""
    missing = [exe for exe in (FFMPEG, FFPROBE) if shutil.which(exe) is None]
    if missing:
        raise SystemExit(
            f"找不到 {', '.join(missing)}，请先安装 FFmpeg 并加入 PATH\n"
            f"    winget install --id Gyan.FFmpeg -e\n"
            f"（详见 TOOLS_SETUP.md）"
        )


def run_ffmpeg(cmd):
    """执行 ffmpeg 命令；失败时把 stderr 一并抛出，便于定位问题。"""
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 执行失败（退出码 {}）:\n{}\n{}".format(
                proc.returncode, " ".join(cmd), proc.stderr.strip())
        )


# ---------------------------------------------------------------------------
# 1. 读取与处理区间
# ---------------------------------------------------------------------------
def read_regions(path):
    """读取 txt，返回 [(start, end), ...]，只取每行前两列（单位秒）。"""
    regions = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = [p for p in re.split(r"[,\s]+", text) if p]
            if len(parts) < 2:
                print(f"  ! 第 {lineno} 行不足两列，已跳过: {text!r}")
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                print(f"  ! 第 {lineno} 行不是数字，已跳过: {text!r}")
                continue
            if end <= start:
                print(f"  ! 第 {lineno} 行 end <= start（{start} ~ {end}），已跳过")
                continue
            regions.append((start, end))
    return regions


def apply_offsets(regions, offset1, offset2, clip_end=None):
    """start -= offset1，end -= offset2，并夹到 [0, clip_end] 内。"""
    adjusted = []
    for start, end in regions:
        s = start - offset1
        e = end - offset2

        if s < 0:
            print(f"  ! {start:.3f} - {offset1:g} < 0，开始时间夹到 0")
            s = 0.0
        if clip_end is not None and e > clip_end:
            print(f"  ! {end:.3f} - {offset2:g} 超出视频时长，结束时间夹到 {clip_end:.3f}")
            e = clip_end

        if e - s <= 0:
            print(f"  ! 区间 [{start:.3f}, {end:.3f}] 偏移后为 "
                  f"[{s:.3f}, {e:.3f}]，无效，已跳过")
            continue

        adjusted.append((s, e))
    return adjusted


def apply_duplicate_shift(regions, interval=DUP_INTERVAL, dup_len=DUP_LEN):
    """
    把真实时间（弹幕时间）映射成视频时间。

    视频每 interval 秒会重复 dup_len 秒，重复之后的内容在视频里整体后移；
    第 k 个重复点之后的真实时间 t，对应视频里的 t + dup_len * k。
    这里用 t // interval 估计已经过了几个重复点。
    """
    if interval <= 0 or dup_len <= 0:
        return list(regions)

    shifted = []
    for start, end in regions:
        k_start = int(start // interval)
        k_end = int(end // interval)
        shifted.append((start + dup_len * k_start,
                        end + dup_len * k_end))
    return shifted


def decode_audio_range(path, start, dur, sr=AUDIO_SR):
    """解码一段音频为单声道 float32（范围 [-1, 1]），用于音量分析。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def frame_rms(x, sr=AUDIO_SR, frame=SNAP_FRAME, hop=SNAP_HOP):
    """返回 (每帧中心时间(秒), 每帧 RMS)。"""
    n = max(int(round(frame * sr)), 1)
    h = max(int(round(hop * sr)), 1)
    if x.size < n:
        return np.empty(0), np.empty(0)

    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    centers = (np.arange(rms.size) * h + n / 2.0) / sr
    return centers, rms


def snap_cut(path, t, direction, max_shift=SNAP_MAX_SHIFT, clip_end=None):
    """
    把剪辑点对齐到附近的静音间隙。

    direction = -1 : 开始点，只能在 [t-max_shift, t] 内往前移；
    direction = +1 : 结束点，只能在 [t, t+max_shift] 内往后移。

    优先切到最长的静音段中间；找不到静音时，切到音量最低的帧。
    解码失败或数据不足时原样返回 t。
    """
    if max_shift <= 0:
        return t

    if direction < 0:
        lo, hi = max(0.0, t - max_shift), t
    else:
        lo, hi = t, t + max_shift
    if clip_end is not None:
        hi = min(hi, clip_end)
    if hi <= lo:
        return t

    win_start = max(0.0, lo - SNAP_PAD)
    win_end = hi + SNAP_PAD
    if clip_end is not None:
        win_end = min(win_end, clip_end)
    if win_end <= win_start:
        return t

    try:
        x = decode_audio_range(path, win_start, win_end - win_start)
    except Exception:
        return t

    centers, rms = frame_rms(x)
    if rms.size == 0:
        return t
    centers = centers + win_start

    # 自适应静音阈值：比噪声底高 6dB，且至少比语音峰值低 20dB
    db = 20.0 * np.log10(rms + 1e-9)
    noise = float(np.percentile(db, 10))
    level = float(np.percentile(db, 90))
    thr = max(noise + 6.0, level - 20.0)
    silent = db < thr

    # 当前切点本身就在静音里，不用调整
    k = int(round((t - win_start - SNAP_FRAME / 2.0) / SNAP_HOP))
    k = min(max(k, 0), silent.size - 1)
    if silent[k]:
        return t

    # 找出所有静音段，取最长的一个（长度相同则取离原切点最近的）
    idx = np.flatnonzero(silent)
    runs = []
    if idx.size:
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, idx.size - 1]
        for a, b in zip(starts, ends):
            t0 = float(centers[idx[a]]) - SNAP_FRAME / 2.0
            t1 = float(centers[idx[b]]) + SNAP_FRAME / 2.0
            length = t1 - t0
            if length < SNAP_MIN_SILENCE:
                continue
            if t1 < lo or t0 > hi:
                continue
            cand = min(max((t0 + t1) / 2.0, lo), hi)
            runs.append((length, abs(cand - t), cand))

    if runs:
        runs.sort(key=lambda r: (-r[0], r[1]))
        return runs[0][2]

    # 没有静音：退回到搜索范围内音量最低的帧
    mask = (centers >= lo) & (centers <= hi)
    if not mask.any():
        return t
    cand_centers = centers[mask]
    cand_rms = rms[mask]
    return float(cand_centers[int(np.argmin(cand_rms))])


def merge_overlaps(regions):
    """合并重叠或首尾相接的区间，避免同一段画面被拼两次。"""
    if not regions:
        return []
    ordered = sorted(regions)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# 2. 探测
# ---------------------------------------------------------------------------
def probe_duration(path):
    """返回视频总时长（秒）。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def probe_format(path):
    """返回 (时长秒, 字节数)，用于核对成品与源视频的体积/码率。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration,size",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True)
    duration, size = proc.stdout.split()
    return float(duration), int(size)


# ---------------------------------------------------------------------------
# 3. 切片与拼接
# ---------------------------------------------------------------------------
def cut_segment(src, start, duration, dst):
    """
    用 stream copy 切出一段（不重编码，零画质损失）。

    -ss 放在 -i 之前：快速定位到目标时间之前最近的关键帧，
        因此实际起点可能比 start 早最多一个 GOP（通常 1~2 秒）。
    -t 表示输出时长，比 -to 更不容易受 seek 影响。
    -avoid_negative_ts make_zero：避免分片带负时间戳，concat 更稳。
    """
    run_ffmpeg([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-map", "0",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        dst,
    ])


def concat_segments(parts, dst, list_path):
    """用 concat demuxer 无损拼接分片（不能用 concat 滤镜，那会触发重编码）。"""
    with open(list_path, "w", encoding="utf-8", newline="\n") as f:
        for p in parts:
            # 路径里的单引号需要转义，否则 concat 列表解析会出错
            safe = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-map", "0", "-c", "copy",
    ]
    if dst.lower().endswith(".mp4"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(dst)
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="按 txt 记录的秒区间剪辑并拼接视频（FFmpeg 无损 stream copy）")
    parser.add_argument("-s", "--offset1", type=float, default=DEFAULT_OFFSET1,
                        help=f"每段开始时间减去的秒数，默认 {DEFAULT_OFFSET1:g}")
    parser.add_argument("-e", "--offset2", type=float, default=DEFAULT_OFFSET2,
                        help=f"每段结束时间减去的秒数，默认 {DEFAULT_OFFSET2:g}")
    parser.add_argument("--no-merge", action="store_true",
                        help="不合并重叠区间（默认合并，避免重复画面）")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留临时分片目录，便于排查问题")
    parser.add_argument("--shift-duplicates", dest="shift_duplicates",
                        action="store_true", default=None,
                        help="打开重复片段补偿（按每 DUP_INTERVAL 秒重复 DUP_LEN 秒推后）")
    parser.add_argument("--no-shift-duplicates", dest="shift_duplicates",
                        action="store_false",
                        help="关闭重复片段补偿")
    parser.add_argument("--dup-len", type=float, default=DUP_LEN,
                        help=f"每个重复片段的长度（秒），默认 {DUP_LEN:g}")
    parser.add_argument("--dup-interval", type=float, default=DUP_INTERVAL,
                        help=f"重复片段之间的间隔（秒），默认 {DUP_INTERVAL:g}")
    parser.add_argument("--snap-shift", type=float, default=SNAP_MAX_SHIFT,
                        help=f"剪辑点找静音时最多移动的秒数，默认 {SNAP_MAX_SHIFT:g}")
    parser.add_argument("--no-snap", action="store_true",
                        help="关闭剪辑点音频对齐")
    args = parser.parse_args()

    shift_duplicates = (SHIFT_FOR_DUPLICATES
                        if args.shift_duplicates is None
                        else args.shift_duplicates)
    snap_boundaries = SNAP_BOUNDARIES and not args.no_snap

    require_tools()

    if not os.path.isfile(VIDEO_PATH):
        raise SystemExit(f"找不到视频文件: {VIDEO_PATH}")
    if not os.path.isfile(REGIONS_TXT):
        raise SystemExit(f"找不到区间文件: {REGIONS_TXT}")

    print(f"视频     : {VIDEO_PATH}")
    print(f"区间文件 : {REGIONS_TXT}")
    print(f"偏移     : 开始 -{args.offset1:g}s, 结束 -{args.offset2:g}s")
    if shift_duplicates:
        print(f"重复补偿 : 每 {fmt_hms(args.dup_interval)} 重复 "
              f"{args.dup_len:g}s，剪辑点按真实时间累计推后")
    else:
        print("重复补偿 : 关闭")
    if snap_boundaries:
        print(f"音频对齐 : 开（边界最多移动 {args.snap_shift:g}s 找静音）")
    else:
        print("音频对齐 : 关闭")

    # ---- 读取区间 ----
    raw = read_regions(REGIONS_TXT)
    print(f"\n读取到 {len(raw)} 段区间")
    if not raw:
        raise SystemExit("区间文件里没有有效数据，退出。")

    duration = probe_duration(VIDEO_PATH)
    print(f"视频时长 : {fmt_hms(duration)} ({duration:.1f}s)")

    # ---- 重复片段补偿：真实时间 -> 视频时间 ----
    if shift_duplicates:
        shifted = apply_duplicate_shift(raw, args.dup_interval, args.dup_len)
        changed = sum(1 for a, b in zip(raw, shifted) if a != b)
        if changed:
            print(f"重复补偿 : {changed} 段区间已按累计偏移推后")
        raw = shifted

    # ---- 应用偏移 ----
    regions = apply_offsets(raw, args.offset1, args.offset2, duration)
    if not regions:
        raise SystemExit("偏移后没有有效区间，退出。")
    print(f"偏移后   : {len(regions)} 段，合计 "
          f"{fmt_hms(sum(e - s for s, e in regions))}")

    # ---- 音频对齐：开始点往前、结束点往后找静音 ----
    if snap_boundaries:
        aligned = []
        moved = 0
        for s, e in regions:
            ns = snap_cut(VIDEO_PATH, s, -1, args.snap_shift, duration)
            ne = snap_cut(VIDEO_PATH, e, +1, args.snap_shift, duration)
            if abs(ns - s) > 1e-3 or abs(ne - e) > 1e-3:
                moved += 1
            aligned.append((ns, ne))
        regions = aligned
        if moved:
            print(f"音频对齐 : {moved} 段边界已移动到静音处")
        else:
            print("音频对齐 : 边界已在静音处，未调整")

    # ---- 合并重叠 ----
    if not args.no_merge:
        merged = merge_overlaps(regions)
        if len(merged) != len(regions):
            print(f"合并重叠 : {len(regions)} 段 -> {len(merged)} 段")
        regions = merged

    total = sum(e - s for s, e in regions)
    print(f"最终计划 : {len(regions)} 段，合计 {fmt_hms(total)} ({total:.1f}s)")

    # ---- 输出容器保持与源一致，保证 stream copy 可用 ----
    src_ext = os.path.splitext(VIDEO_PATH)[1].lower() or ".mp4"
    out = OUTPUT_PATH
    if os.path.splitext(out)[1].lower() != src_ext:
        out = os.path.splitext(out)[0] + src_ext
        print(f"提示     : 输出容器改为与源一致的 {src_ext}，以免流不兼容")

    # ---- 逐段切 + 拼接 ----
    tmp_dir = tempfile.mkdtemp(prefix="auto_edit_")
    try:
        parts = []
        print()
        for idx, (start, end) in enumerate(regions, 1):
            dst = os.path.join(tmp_dir, f"seg_{idx:04d}{src_ext}")
            print(f"  [{idx}/{len(regions)}] {fmt_hms(start)} ~ {fmt_hms(end)}"
                  f"  ({end - start:.1f}s)")
            cut_segment(VIDEO_PATH, start, end - start, dst)
            parts.append(dst)

        print(f"\n拼接 {len(parts)} 个分片 -> {out}")
        concat_segments(parts, out, os.path.join(tmp_dir, "segments.txt"))
    finally:
        if args.keep_temp:
            print(f"临时分片保留在: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- 结果核对 ----
    src_dur, src_size = probe_format(VIDEO_PATH)
    out_dur, out_size = probe_format(out)
    src_kbps = src_size * 8 / src_dur / 1000
    out_kbps = out_size * 8 / out_dur / 1000
    diff = abs(out_kbps - src_kbps) / src_kbps * 100

    print("\n" + "=" * 60)
    print(f"源视频  {fmt_hms(src_dur)}  {src_size / 2**20:9.1f} MB  {src_kbps:8.0f} kbps")
    print(f"成品    {fmt_hms(out_dur)}  {out_size / 2**20:9.1f} MB  {out_kbps:8.0f} kbps")
    print(f"时长比  {out_dur / src_dur * 100:6.1f}%      体积比 "
          f"{out_size / src_size * 100:6.1f}%")
    print(f"单位时间码率差异 {diff:.2f}%（stream copy 下应接近 0 = 画质无损）")
    print("=" * 60)
    print(f"完成: {out}")


if __name__ == "__main__":
    main()

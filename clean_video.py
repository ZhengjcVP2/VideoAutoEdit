#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_video.py — 检测直播回放中「每 30 分钟附近」重复出现的音画片段

背景:
    部分直播回放 / 录播在每 30 分钟刻度附近会重复播放一小段内容（音画都会重复一次）。
    本脚本在每条 30 分钟刻度前后各取 1 分钟，先做音频自相似比对，
    再用低分辨率灰度帧确认画面是否高度一致，最后记录「后一段」重复的起止时间。

流程:
    1. 对每条 30 分钟刻度 T (1800s, 3600s, ...)，取窗口 [T-60s, T+60s]；
       因为前面的重复会把后面的内容往后推，越往后的检测点窗口越长
       （第 i 个检测点前后共多 i*5 秒，例如 2 小时处共多 20 秒）；
    2. 音频: 解码成 16k 单声道 -> log-mel 特征 -> 自相似矩阵，
       沿对角线找连续高相似度的区段（即同一段音频出现两次，允许短暂断点）；
       再用波形归一化互相关复核，排除「频谱相似但内容不同」的误报；
    3. 视频: 以 4fps、64x36 灰度取帧，逐帧比对两次出现位置的画面相似度；
    4. 音频与画面都通过 -> 记录后一段重复的 [start, end]。

输出:
    stdout 只有一行时间列表（单位：秒），可直接交给 auto_edit.py：
        [[1802.35, 1838.90], [3601.2, 3625.6]]
    其余日志全部走 stderr，方便管道处理。

导出:
    把下面的 EXPORT_DUPLICATE_CLIPS 改成 True（或命令行加 --export），
    会把最多 MAX_EXPORT_CLIPS 段重复片段切成独立 mp4 放到 EXPORT_DIR。

用法:
    python clean_video.py
    python clean_video.py sample\testvideo_zzz_day1.mp4
    python clean_video.py --export
    python clean_video.py --window 90 --boundary 1800
"""

import os
import sys
import json
import heapq
import shutil
import argparse
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# 输入 / 输出路径（和 auto_edit.py 一样，先写成变量）
# ---------------------------------------------------------------------------
VIDEO_PATH = r"sample\testvideo_zzz_day1.mp4"     # 待检测的视频

# 打开后导出最多 MAX_EXPORT_CLIPS 段重复片段
EXPORT_DUPLICATE_CLIPS = False
MAX_EXPORT_CLIPS = 3
EXPORT_DIR = r"sample\duplicates"
EXPORT_CRF = 20             # 导出片段重新编码的质量（越小越好）
EXPORT_PRESET = "veryfast"

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# ---------------------------------------------------------------------------
# 检测参数
# ---------------------------------------------------------------------------
BOUNDARY_SEC = 30 * 60      # 每 30 分钟检查一次
WINDOW_SEC = 60.0           # 检查时间点左右各 1 分钟
BOUNDARY_EXTRA_SEC = 5.0    # 每往后一个检测点，窗口前后共拉长 5 秒（前后各 2.5 秒）

MIN_DUP_SEC = 1.0           # 至少重复 1 秒才认
MIN_LAG_SEC = 1.0           # 两次出现的最小间隔，避免把相邻帧当重复
AUDIO_SIM_THRESHOLD = 0.90  # 音频 log-mel 余弦相似度阈值（用于搜索候选）
AUDIO_WAVE_THRESHOLD = 0.75 # 波形归一化互相关阈值（用于最终确认）
UNIQUE_SIM_THRESHOLD = 0.80 # 判定「只重复一次」时，其他位置的波形相关上限
WAVE_SEARCH_SEC = 0.25      # 波形对齐时在候选位置附近搜索的范围
VIDEO_SIM_THRESHOLD = 0.90  # 画面灰度余弦相似度阈值
MAX_HOLE_SEC = 0.15         # 连续匹配中允许的最大断点
TOP_CANDIDATES = 6          # 每个窗口最多验证几个音频候选

# 音频特征参数
SR = 16000
N_FFT = 1024                # 64ms @16k
HOP_SEC = 0.010             # 10ms
MATCH_HOP_SEC = 0.050       # 做相似度矩阵时的帧间隔（50ms）
MAX_MATCH_FRAMES = 3000     # 相似度矩阵帧数上限，超了就再合并
N_MELS = 40
RMS_FLOOR = 0.01            # 低于该 RMS 视为静音（≈ -40 dBFS）
MIN_ACTIVE_FRAC = 0.5       # 重复段里至少一半帧要有声音

# 视频比对参数
VIDEO_FPS = 4
VIDEO_W = 64
VIDEO_H = 36


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    """日志走 stderr，保证 stdout 只有时间列表。"""
    print(*args, file=sys.stderr, flush=True)


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
    """执行 ffmpeg 命令；失败时把 stderr 一并抛出。"""
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 执行失败（退出码 {}）:\n{}\n{}".format(
                proc.returncode, " ".join(cmd), proc.stderr.strip())
        )


def merge_overlaps(regions):
    """合并重叠或首尾相接的区间。"""
    if not regions:
        return []
    ordered = sorted(regions)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# 媒体探测与解码
# ---------------------------------------------------------------------------
def probe_media(path):
    """返回 (时长秒, 是否有音频, 是否有视频)。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败：{proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")

    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0

    streams = data.get("streams", []) or []
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    return duration, has_audio, has_video


def decode_audio(path, start, dur):
    """解码一段为 16k 单声道 float32，范围 [-1, 1]。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(SR),
        "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "音频解码失败：" + proc.stderr.decode("utf-8", "replace").strip())
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def decode_video_gray(path, start, dur, fps=VIDEO_FPS, w=VIDEO_W, h=VIDEO_H):
    """解码一段为低分辨率灰度帧，返回 (n, w*h) float32，范围 [0, 1]。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path,
        "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={w}:{h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        log("  视频解码失败：" + proc.stderr.decode("utf-8", "replace").strip())
        return None

    frame_bytes = w * h
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        return None
    arr = np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8)
    return arr.reshape(n, frame_bytes).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 音频特征
# ---------------------------------------------------------------------------
def _mel_filterbank(sr, n_fft, n_mels, fmin=50.0, fmax=None):
    """构造三角 mel 滤波器组，形状 (n_mels, n_fft//2+1)。"""
    fmax = fmax if fmax is not None else sr / 2.0

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


_FB_CACHE = {}


def _get_filterbank():
    key = (SR, N_FFT, N_MELS)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(SR, N_FFT, N_MELS)
    return _FB_CACHE[key]


def audio_features(path, start, dur):
    """返回 (log-mel 特征 (n, n_mels), 每帧 RMS (n,), 波形)；太短则返回 (None, None, None)。"""
    x = decode_audio(path, start, dur)
    if x.size < N_FFT:
        return None, None, None

    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::int(HOP_SEC * SR)]
    n_frames = frames.shape[0]
    if n_frames < 4:
        return None, None, None

    window = np.hanning(N_FFT).astype(np.float32)
    fb_t = _get_filterbank().T
    mel = np.empty((n_frames, N_MELS), dtype=np.float32)

    block = 1024
    for s in range(0, n_frames, block):
        e = min(n_frames, s + block)
        spec = np.abs(np.fft.rfft(frames[s:e] * window, axis=1)) ** 2
        mel[s:e] = spec.astype(np.float32) @ fb_t

    feat = np.log(mel + 1e-8).astype(np.float32)

    # 每帧 RMS（用于过滤纯静音造成的假重复）
    sq = x.astype(np.float64) ** 2
    cs = np.concatenate(([0.0], np.cumsum(sq)))
    offsets = np.arange(n_frames) * int(HOP_SEC * SR)
    rms = np.sqrt(np.maximum(cs[offsets + N_FFT] - cs[offsets], 0.0) / N_FFT)
    return feat, rms.astype(np.float32), x


def normalize_rows(a):
    """逐行去均值 + L2 归一化，行与行之间的点积即余弦相似度。"""
    a = a.astype(np.float32, copy=True)
    a -= a.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    a /= norm
    return a


def pool_mean(a, factor):
    """沿时间轴按 factor 做均值池化。"""
    if factor <= 1:
        return a
    n = a.shape[0] // factor
    if n == 0:
        return a
    a = a[:n * factor]
    return a.reshape(n, factor, *a.shape[1:]).mean(axis=1)


# ---------------------------------------------------------------------------
# 音频重复检测
# ---------------------------------------------------------------------------
def _overlap_ratio(s1, e1, s2, e2):
    inter = min(e1, e2) - max(s1, s2)
    if inter <= 0:
        return 0.0
    return inter / max(min(e1 - s1, e2 - s2), 1)


def find_duplicate_runs(feat, rms, hop, min_len_sec, min_lag_sec,
                        sim_thresh, top_n=TOP_CANDIDATES):
    """
    在特征序列里找「同一段出现两次」。

    返回候选列表，每项:
        {"a0", "a1": 第一次出现的帧区间, "lag": 帧偏移, "mean_sim", "activity"}
    第二段区间为 [a0+lag, a1+lag]。
    """
    n = feat.shape[0]
    min_len = max(int(round(min_len_sec / hop)), 2)
    min_lag = max(int(round(min_lag_sec / hop)), 1)
    if n < min_len + min_lag:
        return []

    f = normalize_rows(feat)
    sim = f @ f.T                          # (n, n) 余弦相似度
    max_hole = max(int(round(MAX_HOLE_SEC / hop)), 1)

    heap = []                              # 只保留得分最高的若干候选
    for d in range(min_lag, n - min_len + 1):
        diag = np.diagonal(sim, d)
        mask = diag >= sim_thresh
        if not mask.any():
            continue

        idx = np.flatnonzero(mask)
        gaps = np.diff(idx)
        breaks = np.flatnonzero(gaps > max_hole + 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, idx.size - 1]

        for s_i, e_i in zip(starts, ends):
            a0 = int(idx[s_i])
            a1 = int(idx[e_i])
            length = a1 - a0 + 1
            if length < min_len:
                continue

            if rms is not None:
                activity = float(np.mean(rms[a0:a1 + 1] > RMS_FLOOR))
                if activity < MIN_ACTIVE_FRAC:
                    continue
            else:
                activity = 1.0

            mean_sim = float(diag[a0:a1 + 1].mean())
            score = length * mean_sim * mean_sim
            heapq.heappush(heap, (score, a0, a1, d, mean_sim, activity))
            if len(heap) > 400:
                heapq.heappop(heap)

    if not heap:
        return []

    candidates = sorted(heap, key=lambda t: t[0], reverse=True)

    # 相邻 lag 会产生大量互相重叠的候选，只保留每个位置最优的一个
    picked = []
    for score, a0, a1, d, mean_sim, activity in candidates:
        b0, b1 = a0 + d, a1 + d
        duplicated = any(
            _overlap_ratio(a0, a1, p[1], p[2]) > 0.5
            or _overlap_ratio(b0, b1, p[1] + p[3], p[2] + p[3]) > 0.5
            for p in picked
        )
        if duplicated:
            continue
        picked.append((score, a0, a1, d, mean_sim, activity))
        if len(picked) >= top_n:
            break

    return [{"a0": a0, "a1": a1, "lag": d, "mean_sim": ms, "activity": act}
            for _, a0, a1, d, ms, act in picked]


# ---------------------------------------------------------------------------
# 视频确认
# ---------------------------------------------------------------------------
def verify_waveform(wave, t1, t2, length, margin=WAVE_SEARCH_SEC):
    """
    用归一化互相关复核两段波形是否真的相同，并给出精确到采样点的第二段起点。

    参数:
        wave   : 整个窗口的 16k 波形
        t1, t2 : 两段在窗口内的起始时间（秒）
        length : 重复段时长（秒）

    返回:
        (最佳相关系数, 第二段精确起点秒)；样本不足时返回 (None, None)。
    """
    n = len(wave)
    i1 = int(round(t1 * SR))
    L = int(round(length * SR))
    i2 = int(round(t2 * SR))
    if L < int(0.3 * SR):
        return None, None

    x = wave[i1:i1 + L]
    if len(x) < L or L == 0:
        return None, None

    m = int(round(margin * SR))
    y0 = max(0, i2 - m)
    y1 = min(n, i2 + L + m)
    y = wave[y0:y1]
    if len(y) < L:
        return None, None

    x = x - x.mean()
    y = y - y.mean()
    nx = x.size
    N = 1
    while N < y.size + nx:
        N <<= 1

    # cc[k] = sum_i y[k+i] * x[i]
    cc = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(x, N)), N)
    count = y.size - nx + 1
    cc = cc[:count]

    sq = np.concatenate(([0.0], np.cumsum(y * y)))
    energy = sq[nx:count + nx] - sq[:count]
    denom = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(x)
    corr = np.divide(cc, denom, out=np.zeros_like(cc), where=denom > 1e-9)

    # 只在期望位置 ±margin 内搜索
    lo = max(0, (i2 - m) - y0)
    hi = min(count - 1, lo + 2 * m)
    sub = corr[lo:hi + 1]
    if sub.size == 0:
        return None, None

    k = int(np.argmax(sub)) + lo
    best_start = (y0 + k) / SR
    return float(corr[k]), best_start


def is_unique_repeat(wave, t1, length, second_start, tol=0.5):
    """
    检查第一段音频是否「只重复了一次」。

    把第一段波形与整个窗口做归一化互相关，排除自身位置和已找到的重复位置后，
    如果还出现其他高分峰，说明这是循环播放的背景音乐而不是一次性重复。
    """
    L = int(round(length * SR))
    i1 = int(round(t1 * SR))
    k2 = int(round(second_start * SR))
    x = wave[i1:i1 + L]
    if L <= 0 or len(x) < L:
        return False

    x = x.astype(np.float64)
    x = x - x.mean()
    y = wave.astype(np.float64)
    y = y - y.mean()

    N = 1
    while N < y.size + x.size:
        N <<= 1
    cc = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(x, N)), N)
    count = y.size - x.size + 1
    cc = cc[:count]

    sq = np.concatenate(([0.0], np.cumsum(y * y)))
    energy = sq[L:count + L] - sq[:count]
    den = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(x)
    corr = np.divide(cc, den, out=np.zeros(count), where=den > 1e-9)

    # 局部极大值
    peaks = corr >= UNIQUE_SIM_THRESHOLD
    peaks[1:] &= corr[1:] >= corr[:-1]
    peaks[:-1] &= corr[:-1] >= corr[1:]
    if peaks.size:
        peaks[0] = False
        peaks[-1] = False

    tol_n = int(round(tol * SR))
    excluded = np.zeros(count, dtype=bool)
    for center in (i1, k2):
        lo = max(0, center - tol_n)
        hi = min(count, center + tol_n + 1)
        excluded[lo:hi] = True

    return not bool(np.any(peaks & ~excluded))


def compare_video(frames_norm, vfps, win_start, first_start, second_start, length):
    """返回 (平均相似度, 10 分位相似度)。"""
    if frames_norm is None or frames_norm.shape[0] == 0:
        return -1.0, -1.0

    n = frames_norm.shape[0]
    i1 = int(round((first_start - win_start) * vfps))
    i2 = int(round((second_start - win_start) * vfps))
    count = max(int(round(length * vfps)), 1)

    best_mean, best_p10 = -1.0, -1.0
    # 两次重复的时间偏移不一定正好落在整帧上，允许 ±1 帧搜索
    for shift in (-1, 0, 1):
        sims = []
        for k in range(count):
            j1 = i1 + k
            j2 = i2 + k + shift
            if 0 <= j1 < n and 0 <= j2 < n:
                sims.append(float(frames_norm[j1] @ frames_norm[j2]))
        if not sims:
            continue
        arr = np.asarray(sims)
        mean = float(arr.mean())
        if mean > best_mean:
            best_mean = mean
            best_p10 = float(np.percentile(arr, 10))
    return best_mean, best_p10


# ---------------------------------------------------------------------------
# 主分析流程
# ---------------------------------------------------------------------------
def analyze(path, duration, has_audio, has_video,
            boundary=BOUNDARY_SEC, window=WINDOW_SEC):
    """返回重复片段列表 [(start, end), ...]（秒，后一段的区间）。"""
    results = []
    if duration <= 0:
        log("无法读取视频时长，退出")
        return results
    if not has_audio:
        log("视频没有音频流，无法做音频比对；如需处理请先确认文件")
        return results
    if not has_video:
        log("视频没有视频流，无法确认画面一致，跳过")
        return results

    boundary_index = 0
    t = boundary
    while t < duration:
        boundary_index += 1
        # 前面的重复会把后面的内容不断后移，所以越往后的检测点窗口越长：
        # 第 i 个检测点前后共多出 i * BOUNDARY_EXTRA_SEC 秒（前后各一半）。
        extra_half = boundary_index * BOUNDARY_EXTRA_SEC / 2.0
        win_start = max(0.0, t - window - extra_half)
        win_end = min(duration, t + window + extra_half)
        win_dur = win_end - win_start
        if win_dur < max(10.0, MIN_DUP_SEC * 2):
            break

        log(f"[{boundary_index:02d}] 检查 {fmt_hms(t)} 附近 "
            f"({fmt_hms(win_start)} ~ {fmt_hms(win_end)}，{win_dur:.1f}s)")

        try:
            feat, rms, wave = audio_features(path, win_start, win_dur)
        except RuntimeError as exc:
            log(f"  音频处理失败：{exc}")
            t += boundary
            continue

        if feat is None:
            log("  音频太短，跳过")
            t += boundary
            continue

        # 控制相似度矩阵规模
        factor = max(1, int(round(MATCH_HOP_SEC / HOP_SEC)))
        if feat.shape[0] // factor > MAX_MATCH_FRAMES:
            factor = int(np.ceil(feat.shape[0] / MAX_MATCH_FRAMES))
        hop = HOP_SEC * factor
        feat_p = pool_mean(feat, factor)
        rms_p = pool_mean(rms.reshape(-1, 1), factor).ravel()

        candidates = find_duplicate_runs(
            feat_p, rms_p, hop, MIN_DUP_SEC, MIN_LAG_SEC, AUDIO_SIM_THRESHOLD)
        if not candidates:
            log("  未发现重复音频")
            t += boundary
            continue

        log(f"  音频候选 {len(candidates)} 个，最佳相似度 "
            f"{candidates[0]['mean_sim']:.3f}")

        frames = decode_video_gray(path, win_start, win_dur)
        frames_norm = normalize_rows(frames) if frames is not None else None

        accepted = None
        for cand in candidates:
            rel_first = cand["a0"] * hop
            rel_second = (cand["a0"] + cand["lag"]) * hop
            length = (cand["a1"] - cand["a0"] + 1) * hop
            first_start = win_start + rel_first
            first_end = first_start + length

            # 波形复核：真正重复的音频互相关很高，仅频谱相似的会被刷掉
            wave_corr, refined_rel_second = verify_waveform(
                wave, rel_first, rel_second, length)
            if wave_corr is None or wave_corr < AUDIO_WAVE_THRESHOLD:
                shown = "n/a" if wave_corr is None else f"{wave_corr:.3f}"
                log(f"  候选 前段 {fmt_hms(first_start)}~{fmt_hms(first_end)} "
                    f"波形相关 {shown} -> 不是同一段音频")
                continue

            second_start = win_start + refined_rel_second
            second_end = second_start + length

            if not is_unique_repeat(wave, rel_first, length, refined_rel_second):
                log(f"  候选 前段 {fmt_hms(first_start)}~{fmt_hms(first_end)} "
                    f"在窗口内出现不止两次（疑似循环音乐），跳过")
                continue

            if second_end > duration + 0.5:
                log("  候选超出视频时长，跳过")
                continue
            if frames_norm is None:
                log("  没有可用视频帧，无法确认画面")
                break

            mean_sim, p10 = compare_video(
                frames_norm, VIDEO_FPS, win_start,
                first_start, second_start, length)
            ok = mean_sim >= VIDEO_SIM_THRESHOLD
            log(f"  候选 前段 {fmt_hms(first_start)}~{fmt_hms(first_end)}，"
                f"后段 {fmt_hms(second_start)}~{fmt_hms(second_end)}，"
                f"波形={wave_corr:.3f}，画面 mean={mean_sim:.3f} "
                f"p10={p10:.3f} -> {'通过' if ok else '不通过'}")

            if ok:
                accepted = (second_start, second_end)
                break

        if accepted:
            results.append(accepted)
            log(f"  ==> 标记重复：{fmt_hms(accepted[0])} ~ {fmt_hms(accepted[1])}"
                f"（{accepted[1] - accepted[0]:.2f}s）")

        t += boundary

    return merge_overlaps(results)


# ---------------------------------------------------------------------------
# 导出片段
# ---------------------------------------------------------------------------
def export_clips(src, segments, out_dir=EXPORT_DIR, limit=MAX_EXPORT_CLIPS):
    """
    导出最多 limit 段重复片段。

    这里使用输入侧 -ss + 重新编码，原因是 stream copy 只能从关键帧开始，
    会把起点之前最多一个 GOP 的内容一起带进来。片段通常只有几秒，重编码很快。
    """
    if not segments:
        return []
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, (start, end) in enumerate(segments[:limit], 1):
        name = "dup_{:02d}_{}__{}.mp4".format(
            i,
            fmt_hms(start).replace(":", "-"),
            fmt_hms(end).replace(":", "-"))
        dst = os.path.join(out_dir, name)
        run_ffmpeg([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}",
            "-i", src,
            "-t", f"{end - start:.3f}",
            "-c:v", "libx264",
            "-preset", EXPORT_PRESET,
            "-crf", str(EXPORT_CRF),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            dst,
        ])
        saved.append(dst)
        log(f"  已导出：{dst}")
    return saved


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="检测直播回放中每 30 分钟附近的音画重复片段")
    parser.add_argument("video", nargs="?", default=VIDEO_PATH, help="待检测视频")
    parser.add_argument("--boundary", type=float, default=BOUNDARY_SEC,
                        help="检查间隔，默认 1800 秒")
    parser.add_argument("--window", type=float, default=WINDOW_SEC,
                        help="时间点左右各检查多少秒，默认 60")
    parser.add_argument("--export", action="store_true",
                        help="导出最多 3 段重复片段")
    parser.add_argument("--export-dir", default=EXPORT_DIR, help="导出目录")
    args = parser.parse_args()

    require_tools()

    duration, has_audio, has_video = probe_media(args.video)
    log(f"视频：{args.video}")
    log(f"时长：{fmt_hms(duration)}（{duration:.1f}s），"
        f"音频={has_audio}，视频={has_video}")

    segments = analyze(args.video, duration, has_audio, has_video,
                       boundary=args.boundary, window=args.window)
    segments = [(round(float(s), 2), round(float(e), 2)) for s, e in segments]

    if args.export or EXPORT_DUPLICATE_CLIPS:
        export_clips(args.video, segments, args.export_dir)

    if not segments:
        log("没有检测到音画重复片段")
    else:
        log("检测结果（后一段重复的起止时间）：")
        for s, e in segments:
            log(f"  {fmt_hms(s)} ~ {fmt_hms(e)}    [{s:.2f}, {e:.2f}]")

    # stdout 只输出时间列表
    print(json.dumps([[s, e] for s, e in segments]))


if __name__ == "__main__":
    main()

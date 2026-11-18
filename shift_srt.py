#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shift_srt.py — 把 SRT 字幕整体平移指定秒数（默认 +55 秒）

特点:
    - 只改时间轴，字幕文本、行数、顺序原样保留
    - 支持正数（整体向后/延后）和负数（整体向前/提前）
    - 平移后早于 0 秒的字幕会被裁剪到 0 秒，完全落在 0 秒前的会丢弃并提示
    - 自动重新编号，兼容 CRLF / BOM / 逗号或点作为毫秒分隔符
    - 支持一次传多个文件，或传文件夹批量处理（--suffix 控制输出文件名）

用法:
    python shift_srt.py "xxx.srt"                       # 默认整体 +55 秒
    python shift_srt.py "xxx.srt" -s 30                 # 整体 +30 秒
    python shift_srt.py "xxx.srt" -s -12.5              # 整体 -12.5 秒
    python shift_srt.py "xxx.srt" -o out.srt            # 指定输出文件
    python shift_srt.py "xxx.srt" --in-place            # 直接覆盖原文件（先备份 .bak）
    python shift_srt.py video_source\文七星铁3测 day3-4  # 文件夹批量处理
    python shift_srt.py "xxx.srt" -s 55 --dry-run       # 只看结果不写文件
"""

import os
import re
import sys
import shutil
import argparse

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_SHIFT = 55.0            # 默认平移秒数（正数 = 向后/延后）
DEFAULT_SUFFIX = "_shift"       # 批量/默认输出文件名后缀

TIME_RE = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


# ---------------------------------------------------------------------------
# 时间戳工具
# ---------------------------------------------------------------------------
def parse_time(h, m, s, ms):
    """时间戳分量 -> 秒（浮点）。"""
    ms = (ms + "000")[:3]          # 兼容 1~3 位毫秒
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def fmt_srt_time(seconds):
    """秒 -> SRT 时间戳 HH:MM:SS,mmm。"""
    ms = max(int(round(float(seconds) * 1000)), 0)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# 解析 / 写回
# ---------------------------------------------------------------------------
def parse_srt(text):
    """解析 SRT -> [(start_sec, end_sec, 文本)]，跳过无时间轴的块。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip("\n")
        if not block.strip():
            continue
        lines = block.split("\n")
        # 第一行可能是序号，也可能是直接的时间轴
        idx = 0
        if idx < len(lines) and not TIME_RE.search(lines[idx]) and lines[idx].strip().isdigit():
            idx += 1
        if idx >= len(lines):
            continue
        m = TIME_RE.search(lines[idx])
        if not m:
            continue
        g = m.groups()
        start = parse_time(*g[0:4])
        end = parse_time(*g[4:8])
        body = "\n".join(lines[idx + 1:]).strip("\n")
        cues.append((start, end, body))
    return cues


def shift_cues(cues, offset):
    """整体平移；返回 (新 cues, 被丢弃条数, 被裁剪条数)。"""
    shifted, dropped, clamped = [], 0, 0
    for start, end, body in cues:
        ns, ne = start + offset, end + offset
        if ne <= 0:
            dropped += 1
            continue
        if ns < 0:
            ns, clamped = 0.0, clamped + 1
        shifted.append((ns, ne, body))
    return shifted, dropped, clamped


def render_srt(cues):
    """cues -> SRT 文本（重新编号，CRLF 换行）。"""
    out = []
    for i, (start, end, body) in enumerate(cues, 1):
        out.append(f"{i}\r\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\r\n{body}\r\n")
    return "\r\n".join(out)


# ---------------------------------------------------------------------------
# 文件处理
# ---------------------------------------------------------------------------
def process_file(path, offset, out_path=None, in_place=False,
                 suffix=DEFAULT_SUFFIX, dry_run=False):
    """处理单个 SRT 文件，返回输出路径（dry-run 时返回 None）。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        cues = parse_srt(f.read())

    if not cues:
        print(f"[跳过] 没解析到任何字幕：{path}")
        return None

    shifted, dropped, clamped = shift_cues(cues, offset)
    if not shifted:
        print(f"[跳过] 平移 {offset:+.3f}s 后所有字幕都在 0 秒之前：{path}")
        return None

    text = render_srt(shifted)
    sign = "+" if offset >= 0 else "-"

    if dry_run:
        print(f"[预览] {path}  平移 {sign}{abs(offset):g}s  "
              f"{len(cues)} 条 -> {len(shifted)} 条")
        for i, (s, e, body) in enumerate(shifted[:5], 1):
            print(f"    {i}  {fmt_srt_time(s)} --> {fmt_srt_time(e)}  "
                  f"{body.splitlines()[0] if body else ''}")
        if len(shifted) > 5:
            print(f"    ... 共 {len(shifted)} 条")
        return None

    if in_place:
        if out_path:
            raise SystemExit("--in-place 与 -o 不能同时使用")
        backup = path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        out_path = path
    elif not out_path:
        stem, _ = os.path.splitext(path)
        out_path = f"{stem}{suffix}{sign}{abs(offset):g}s.srt"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    extra = []
    if clamped:
        extra.append(f"{clamped} 条起点被裁到 0 秒")
    if dropped:
        extra.append(f"{dropped} 条被丢弃")
    extra_txt = f"（{', '.join(extra)}）" if extra else ""
    print(f"[完成] {path}  平移 {sign}{abs(offset):g}s  "
          f"{len(cues)} 条 -> {len(shifted)} 条{extra_txt}  ->  {out_path}")
    return out_path


def collect_inputs(paths):
    """展开文件夹里的 .srt 文件，保持传入顺序。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(
                os.path.join(p, n) for n in os.listdir(p)
                if n.lower().endswith(".srt")
            )
            if not found:
                print(f"[跳过] 文件夹里没有 .srt：{p}")
            files.extend(found)
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[跳过] 找不到：{p}")
    return files


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="把 SRT 字幕整体平移指定秒数（正数向后，负数向前）")
    ap.add_argument("inputs", nargs="+", help="一个或多个 .srt 文件，或包含 .srt 的文件夹")
    ap.add_argument("-s", "--shift", type=float, default=DEFAULT_SHIFT,
                    help=f"平移秒数，正=向后、负=向前（默认 {DEFAULT_SHIFT:g}）")
    ap.add_argument("-o", "--output", help="输出文件（仅单文件输入时有效）")
    ap.add_argument("--in-place", action="store_true",
                    help="直接覆盖原文件（首次会生成 .bak 备份）")
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX,
                    help=f"默认输出文件名后缀（默认 {DEFAULT_SUFFIX!r}）")
    ap.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        raise SystemExit("没有可处理的 .srt 文件")

    if args.output and len(files) > 1:
        raise SystemExit("-o 只能配合单个输入文件使用")

    for path in files:
        process_file(path, args.shift,
                     out_path=args.output,
                     in_place=args.in_place,
                     suffix=args.suffix,
                     dry_run=args.dry_run)


if __name__ == "__main__":
    main()

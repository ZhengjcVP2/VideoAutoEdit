#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_titles.py — 从剪辑计划 txt（*.plan.txt）里只抽出「标题」行

特点:
    - 只保留含有「标题」的行，丢掉时间轴/理由/台词/音频对齐等其它行
    - 去掉「标题:」前缀和行首缩进，只留标题正文
    - 保持原有顺序，可选去重、可选加编号
    - 支持一次传多个文件，或传文件夹批量处理
    - 输出 `<原名>_titles.txt`（与原文件同目录），兼容 BOM / CRLF / LF

用法:
    python extract_titles.py "video_source\\文七绝区零day3\\export\\文七绝区零day3_edited.plan.txt"
    python extract_titles.py "xxx.plan.txt" -o titles.txt     # 指定输出文件
    python extract_titles.py "xxx.plan.txt" --with-index      # 带 01 02 编号
    python extract_titles.py "xxx.plan.txt" --dedupe          # 重复标题只留一个
    python extract_titles.py video_source                     # 文件夹批量处理
    python extract_titles.py "xxx.plan.txt" --dry-run         # 只打印不写文件
"""

import os
import re
import sys
import argparse

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_SUFFIX = "_titles"          # 输出文件名后缀
KEYWORD = "标题"                     # 要抓取的关键字
KEY_STRIP_RE = re.compile(r"^[\s:：\-—、.]+")   # 「标题」后面的分隔符

# 批量处理时跳过的东西
SKIP_SUFFIXES = ("_titles.txt",)


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
def extract_title(line):
    """一行 -> 标题正文；不含关键字返回 None。"""
    idx = line.find(KEYWORD)
    if idx < 0:
        return None
    rest = line[idx + len(KEYWORD):]
    return KEY_STRIP_RE.sub("", rest).strip()


def extract_titles(text, dedupe=False):
    """整段文本 -> 标题列表（保持原顺序）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    titles = []
    seen = set()
    for line in text.split("\n"):
        title = extract_title(line)
        if not title:
            continue
        if dedupe:
            if title in seen:
                continue
            seen.add(title)
        titles.append(title)
    return titles


def format_titles(titles, with_index=False, sep="\n"):
    """标题列表 -> 输出文本。"""
    if not with_index:
        return sep.join(titles) + (sep if titles else "")
    width = max(2, len(str(len(titles))))
    rows = [f"{i:0{width}d} {t}" for i, t in enumerate(titles, 1)]
    return sep.join(rows) + (sep if rows else "")


# ---------------------------------------------------------------------------
# 文件 / 批量
# ---------------------------------------------------------------------------
def default_output(path):
    """输入路径 -> 默认输出路径（同目录，加后缀）。"""
    stem, _ = os.path.splitext(path)
    return stem + DEFAULT_SUFFIX + ".txt"


def collect_inputs(targets):
    """参数列表 -> 实际要处理的 txt 文件列表。"""
    files = []
    for target in targets:
        if os.path.isdir(target):
            for name in sorted(os.listdir(target)):
                if not name.lower().endswith(".txt"):
                    continue
                if name.lower().endswith(SKIP_SUFFIXES):
                    continue
                files.append(os.path.join(target, name))
        elif os.path.isfile(target):
            files.append(target)
        else:
            print(f"[跳过] 找不到: {target}", file=sys.stderr)
    return files


def process_file(path, out_path, with_index=False, dedupe=False, dry_run=False):
    """处理单个文件，返回写出的标题条数。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    titles = extract_titles(text, dedupe=dedupe)
    result = format_titles(titles, with_index=with_index)

    print(f"[处理] {path} -> {len(titles)} 条标题")
    if dry_run:
        print(result, end="")
        return len(titles)

    with open(out_path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(result)
    print(f"[写出] {out_path}")
    return len(titles)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="从剪辑计划 txt 里只抽出含有「标题」的行（去掉「标题:」前缀）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="计划 txt 文件，或文件夹（批量处理里面的 .txt）")
    parser.add_argument("-o", "--output", help="输出文件（只在单个输入时有效）")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX,
                        help=f"默认输出文件名后缀（默认 {DEFAULT_SUFFIX}）")
    parser.add_argument("--with-index", action="store_true", help="每行前面加序号")
    parser.add_argument("--dedupe", action="store_true", help="重复标题只保留第一次出现的")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写文件")
    args = parser.parse_args(argv)

    files = collect_inputs(args.inputs)
    if not files:
        print("[错误] 没有可处理的 txt 文件", file=sys.stderr)
        return 1

    if args.output and len(files) > 1:
        print("[警告] 传了多个输入，忽略 -o，按默认后缀分别输出", file=sys.stderr)
        args.output = None

    total = 0
    for path in files:
        out_path = args.output or (os.path.splitext(path)[0] + args.suffix + ".txt")
        try:
            total += process_file(path, out_path, with_index=args.with_index,
                                 dedupe=args.dedupe, dry_run=args.dry_run)
        except OSError as exc:
            print(f"[失败] {path}: {exc}", file=sys.stderr)
            continue

    print(f"[完成] {len(files)} 个文件，共 {total} 条标题")
    return 0


if __name__ == "__main__":
    sys.exit(main())

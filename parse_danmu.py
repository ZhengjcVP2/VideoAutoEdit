#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站弹幕密度图生成器

支持格式:
    - .xml : 标准B站弹幕文件 <d p="时间,模式,字号,颜色,...">文本</d>
    - .txt / 其它 : 尝试用正则从中提取 p="..." 或行首时间

用法:
    python danmaku_density.py danmaku.xml
    python danmaku_density.py danmaku.xml -n 30 -o density.png
"""

import os
import re
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm


# ---------------------------------------------------------------------------
# 0. 可调参数
# ---------------------------------------------------------------------------
HIGHLIGHT_COLOR = '#B3E5FC'   # 弹幕密集段的淡蓝色背景
HIGHLIGHT_ALPHA = 0.80


# ---------------------------------------------------------------------------
# 1. 解析
# ---------------------------------------------------------------------------
def parse_xml(path):
    """解析标准 B 站 xml 弹幕文件，返回每一条弹幕的时间（单位：秒）。"""
    times = []
    # 使用 iterparse 逐条处理，避免大文件一次性载入内存
    for _event, elem in ET.iterparse(path, events=('end',)):
        if elem.tag == 'd':
            p = elem.get('p')
            if p:
                try:
                    # p 属性第一段即为时间
                    times.append(float(p.split(',', 1)[0]))
                except ValueError:
                    pass
            elem.clear()
    return times


def parse_text(path):
    """从 txt / json 等文本格式中提取时间。
    优先匹配 <d p="..."> 结构；否则退化为行首第一个数字。"""
    times = []
    re_p   = re.compile(r'p="([^"]+)"')
    re_num = re.compile(r'^\s*(-?\d+(?:\.\d+)?)')

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re_p.search(line)
            if m:
                try:
                    times.append(float(m.group(1).split(',', 1)[0]))
                    continue
                except ValueError:
                    pass
            m2 = re_num.match(line)
            if m2:
                try:
                    times.append(float(m2.group(1)))
                except ValueError:
                    pass
    return times


def load_times(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xml':
        return parse_xml(path)
    return parse_text(path)


# ---------------------------------------------------------------------------
# 2. 滑动窗口统计
# ---------------------------------------------------------------------------
def sliding_window(times, window=20.0, step=None):
    """
    对排序后的 times 做滑动窗口统计。

    参数
    ----
    times  : 迭代，弹幕时间（秒）
    window : 窗口宽度（秒），默认 20
    step   : 窗口滑动步长（秒），默认 window / 10

    返回
    ----
    centers : 窗口中心时刻数组（秒）
    counts  : 对应窗口内的弹幕数量
    """
    if not times:
        return np.array([]), np.array([])

    times = np.sort(np.asarray(times, dtype=float))
    t_max = float(times[-1])
    if step is None:
        step = max(window / 10.0, 0.5)

    centers = np.arange(0.0, t_max + step, step)

    # 左闭右闭区间 [c - w/2, c + w/2]
    lo = np.searchsorted(times, centers - window / 2.0, side='left')
    hi = np.searchsorted(times, centers + window / 2.0, side='right')
    counts = (hi - lo).astype(int)

    return centers, counts


def find_high_density_intervals(centers, counts, threshold, window):
    """
    找出弹幕数超过阈值的连续时间段（把相邻的密集窗口合并成一段）。

    参数
    ----
    centers   : 窗口中心时刻数组（秒）
    counts    : 每个窗口内的弹幕数量
    threshold : 阈值，严格大于该值才算“密集”
    window    : 窗口宽度（秒），用于把窗口中心还原成时间区间

    返回
    ----
    intervals : [(start, end), ...]，单位秒，按时间升序且互不重叠
    """
    n = len(counts)
    if n == 0:
        return []

    half = window / 2.0
    intervals = []
    i = 0
    while i < n:
        if counts[i] <= threshold:
            i += 1
            continue
        # 向后扩展，直到下一个窗口不再密集
        j = i
        while j + 1 < n and counts[j + 1] > threshold:
            j += 1
        start = max(float(centers[i]) - half, 0.0)
        end = max(float(centers[j]) + half, start)
        intervals.append((start, end))
        i = j + 1
    return intervals


def analyze_density(times, window=20.0, step=None, threshold=10.0):
    """一次算好窗口中心、窗口计数和超过阈值的连续时间段。"""
    centers, counts = sliding_window(times, window=window, step=step)
    intervals = find_high_density_intervals(centers, counts, threshold, window)
    return centers, counts, intervals


def default_analysis_path(src_path):
    """由源文件名推出 '<原文件名>_analyzed.txt'（与原文件放在同一目录）。"""
    stem, _ext = os.path.splitext(src_path)
    return f'{stem}_analyzed.txt'


def write_intervals(intervals, path):
    """
    把弹幕高频区域写入 txt，每行对应一个区域，三列（单位：秒，制表符分隔）：

        start<TAB>end<TAB>duration

    文件已存在时直接覆盖。返回 (写入路径, 是否覆盖了旧文件)。
    """
    existed = os.path.exists(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for start, end in intervals:
            f.write(f'{start:.3f}\t{end:.3f}\t{end - start:.3f}\n')

    return path, existed


# ---------------------------------------------------------------------------
# 3. 绘图
# ---------------------------------------------------------------------------
def _setup_chinese_font():
    """尽量让 matplotlib 正常显示中文。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ('SimHei', 'Microsoft YaHei', 'PingFang SC',
                 'Heiti SC', 'Hiragino Sans GB', 'Noto Sans CJK SC',
                 'Source Han Sans SC', 'WenQuanYi Micro Hei'):
        if name in available:
            matplotlib.rcParams['font.sans-serif'] = [name]
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


def _pick_time_tick(t_max, target_ticks=10):
    """根据总时长选择一组合适的 x 轴刻度间隔（秒）。"""
    for c in (10, 15, 30, 60, 120, 300, 600, 900, 1800,
              3600, 7200, 10800, 21600, 43200, 86400):
        if t_max / c <= target_ticks:
            return c
    return 86400


def _fmt_hhmm(x_seconds, _pos=None):
    """把秒转为 'HH:MM' 显示。"""
    x = int(max(x_seconds, 0))
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}"


def _fmt_hms(x_seconds):
    """把秒转为 'HH:MM:SS' 显示，用于终端输出时长。"""
    x = int(max(round(float(x_seconds)), 0))
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def plot_density(times, window=20.0, step=None, output=None, threshold=10.0,
                 intervals=None):
    _setup_chinese_font()

    if not times:
        print('未解析到任何弹幕，无法绘图。')
        return

    centers, counts = sliding_window(times, window=window, step=step)
    actual_step = (float(centers[1] - centers[0]) if len(centers) > 1
                   else float(window))

    # 找出弹幕数超过阈值的连续时间段（调用方算过就直接复用）
    if intervals is None:
        intervals = find_high_density_intervals(centers, counts,
                                                threshold, window)
    total_seconds = sum(end - start for start, end in intervals)

    # ------------------------- 终端输出 -------------------------
    if intervals:
        print(f'\n弹幕数超过阈值 {threshold:g} 条的密集段共 {len(intervals)} 处：')
        for idx, (start, end) in enumerate(intervals, 1):
            print(f'  {idx:>3}. {_fmt_hms(start)} ~ {_fmt_hms(end)}'
                  f'  时长 {_fmt_hms(end - start)}')
        print(f'密集段总时长: {_fmt_hms(total_seconds)}'
              f'（{total_seconds:.1f} 秒）')
    else:
        print(f'\n没有弹幕数超过阈值 {threshold:g} 条的片段。')

    # ------------------------- 绘图 -------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    # 先用淡蓝色长方形铺底，标记密集时间段
    for k, (start, end) in enumerate(intervals):
        ax.axvspan(start, end,
                   color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA,
                   linewidth=0, zorder=0,
                   label=(f'弹幕数 > {threshold:g}' if k == 0 else None))

    ax.fill_between(centers, counts, color='tab:blue', alpha=0.30,
                    linewidth=0, zorder=2)
    ax.plot(centers, counts, color='tab:blue', linewidth=1.0, zorder=3)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.set_axisbelow(True)          # 网格线画在图形下方

    ax.set_xlabel('时间（小时:分钟）')
    ax.set_ylabel(f'每 {window:g} 秒弹幕数')

    title = (f'弹幕密度（窗口 {window:g}s，步长 {actual_step:g}s，'
             f'共 {len(times)} 条）')
    if intervals:
        title += f'\n超过 {threshold:g} 条的密集段总时长 {_fmt_hms(total_seconds)}'
    ax.set_title(title)

    # x 轴刻度：HH:MM
    tick_step = _pick_time_tick(centers[-1] if len(centers) else 0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_hhmm))

    # 阈值参考线
    ax.axhline(threshold, color='tab:red', linestyle='--',
               linewidth=1.0, alpha=0.7, zorder=1,
               label=f'阈值 {threshold:g}')

    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f'图片已保存: {output}')

    plt.show()


# ---------------------------------------------------------------------------
# 4. 命令行入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='根据 B 站弹幕文件绘制弹幕密度图')
    #parser.add_argument('file', help='弹幕文件路径（.xml / .txt 等）')
    parser.add_argument('-n', '--window', type=float, default=30.0,
                        help='滑动窗口大小（秒），默认 30')
    parser.add_argument('-s', '--step', type=float, default=None,
                        help='滑动步长（秒），默认 window/10')
    parser.add_argument('-t', '--threshold', type=float, default=12.0,
                        help='弹幕数超过该值的片段用淡蓝色背景高亮，默认 12')
    parser.add_argument('-o', '--output', default=None,
                        help='图片输出路径；不指定则只弹窗显示')
    parser.add_argument('-a', '--analyze-output', default=None,
                        help='高频区域 txt 输出路径，默认 <原文件名>_analyzed.txt')
    args = parser.parse_args()

    src = r"sample\test1.xml"
    times = load_times(src)
    print(f'共解析到 {len(times)} 条弹幕')
    if times:
        print(f'时间范围: {_fmt_hhmm(min(times))} ~ {_fmt_hhmm(max(times))}')

    # 计算高频区域，并写入 '<原文件名>_analyzed.txt'（存在则覆盖）
    _centers, _counts, intervals = analyze_density(
        times, window=args.window, step=args.step, threshold=args.threshold)

    analyzed_path = args.analyze_output or default_analysis_path(src)
    path, overwritten = write_intervals(intervals, analyzed_path)
    verb = '已覆盖旧文件' if overwritten else '已创建'
    print(f'高频区域 txt {verb}: {path}（{len(intervals)} 行）')

    plot_density(times, window=args.window, step=args.step,
                 output=args.output, threshold=args.threshold,
                 intervals=intervals)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ASR_and_cut.py — 一条命令跑完：文件夹内所有视频 -> 逐片字幕 -> 自动组合剪辑

本文件只做「参数 + 串联」，算法全部留在被调用的脚本里：
    第 1 步  sensevoice_srt.py   每个分片生成同名 SRT（本地 SenseVoice），
            默认再用大模型做一轮字幕纠错（同音字 / 游戏术语 / 人名）
    第 2 步  edit_with_caption.py --video-dir <文件夹>
            把分片按文件名顺序拼成一条全局时间轴，交给大模型挑片段并渲染

输出位置:
    分片的 ASR 字幕留在视频旁边（<分片名>.srt，配对用）；
    成片、剪辑计划（.plan.json / .plan.txt）、成片字幕（.srt）统一写到
    <输入文件夹>/export/ 。以前写在文件夹上一级的计划仍然会被识别使用。

两个脚本都是用「当前这个 Python（sys.executable）」以子进程方式调用的，
所以虚拟环境、torch / funasr 依赖、控制台中文输出都与直接运行一致；
子进程的进度会实时打印到同一个窗口。

视频 / 字幕的排序、配对、时间轴偏移、剪辑点对齐等逻辑，都复用
edit_with_caption.py 里的 list_files / natural_key / build_parts 等函数，
本文件不重复实现。

用法:
    python run_ASR_and_cut.py                       # 用下面 VIDEO_DIR 常量
    python run_ASR_and_cut.py "video_source\\xxx"    # 指定文件夹
    python run_ASR_and_cut.py --asr-only            # 只生成字幕
    python run_ASR_and_cut.py --cut-only            # 只用现成字幕剪辑
    python run_ASR_and_cut.py --force-asr           # 已有字幕也重新识别
    python run_ASR_and_cut.py --no-correct          # 不做字幕纠错（少一轮大模型）
    python run_ASR_and_cut.py --limit-minutes 3     # 调试：每个分片只识别前 3 分钟
    python run_ASR_and_cut.py --dry-run             # 只出剪辑计划，不渲染
    python run_ASR_and_cut.py --prompt "重点找眼神不好的吐槽"
    python run_ASR_and_cut.py --no-plan             # 忽略现成计划，重新让大模型挑
    python run_ASR_and_cut.py --plan "某个.plan.json"   # 指定计划文件

已有剪辑计划（cut 步骤）:
    剪辑端每次都会把计划写成 <成片名>.plan.json / .plan.txt（在 export 里）。
    下次再跑时，只要这个文件还在，就会自动跳过「分块分析 -> 全局挑选」，
    直接按计划里的时间段渲染（ASR 与字幕纠错仍按原设置跑，用于生成成片字幕）。
    想重新分析：--no-plan；想换一份计划：--plan <路径>。

字幕纠错（ASR 步骤，默认开）:
    sensevoice_srt.py 每次识别完会调大模型修字幕（同音字 / 游戏术语 / 人名），
    需要环境变量 DEEPSEEK_API_KEY；没配时自动跳过，可用 --no-correct 关闭。
    已有字幕默认跳过，所以想重新纠错要加 --force-asr。

主播声纹参考样本（字幕里的「（主播）」标注靠它）:
    python run_ASR_and_cut.py --streamer-ref "样本\\主播说话.mp4"
    python run_ASR_and_cut.py --streamer-ref "样本\\主播.wav" --ref-sec 60
    # 视频 / 音频都行，视频会自动抽音轨；不给则让 sensevoice_srt.py 自动注册
    # 也可以直接把 ASR_STREAMER_REF 常量写成样本路径，之后不带参数就跑
"""

import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 只借用「识别哪些文件 + 怎么排序」的规则，和剪辑端保持完全一致
from edit_with_caption import (  # noqa: E402
    EXPORT_DIR_NAME, VIDEO_EXTS, list_files,
)

# ---------------------------------------------------------------------------
# 输入 / 输出
# ---------------------------------------------------------------------------
VIDEO_DIR = r"video_source\文七原神9月day12-13"
OUTPUT_PATH = ""            # 空 = <文件夹>\export\<文件夹名>_edited.mp4

# ---------------------------------------------------------------------------
# 第 1 步：字幕参数（对应 sensevoice_srt.py）
# ---------------------------------------------------------------------------
ASR_ENABLED = True
ASR_SKIP_EXISTING = True    # 已经有非空 SRT 就跳过
ASR_DEVICE = "auto"         # auto / cuda:0 / cpu
ASR_LANGUAGE = "zh"         # zh / en / yue / ja / ko / auto
ASR_LIMIT_MINUTES = 0.0     # 0 = 全部；调试时设为几 分钟
ASR_MARK_STREAMER = True    # 字幕里标注（主播）

# 主播声纹参考样本：一段「主播在说话」的视频或音频
#   音频：wav / mp3 / m4a …；视频：mp4 / flv / mkv …（自动抽音轨）
#   留空 = 让 sensevoice_srt.py 自动注册（按「贯穿整场的那类声音」推主播）
ASR_STREAMER_REF = r"video_source\sample voice\文七3.2视频.mp4"
ASR_STREAMER_THRESHOLD = 0.0  # 0 = 用 sensevoice_srt.py 的默认阈值（0.45）
ASR_REF_SEC = 0.0             # 参考样本最多用前多少秒，0 = 用默认值（300）

ASR_MAX_HOURS = 0.0         # 0 = 用 sensevoice_srt.py 的默认上限（9 小时）
ASR_CORRECT = True          # 默认开：识别完用大模型修字幕（需要 DEEPSEEK_API_KEY）
ASR_EXTRA_ARGS = []         # 其余参数原样追加，例如 ["--no-punc"]

# ---------------------------------------------------------------------------
# 第 2 步：剪辑参数（对应 edit_with_caption.py）
# ---------------------------------------------------------------------------
CUT_ENABLED = True
CUT_USE_DANMAKU = False     # 使用弹幕
CUT_PROMPT = r"""如果含有小心海或者300朵清心的内容，可以优先考虑。
务必包含以下内容：QQ人，因为我爸管QQ叫丘丘。
我要是误会他了，我给他道歉。走在提瓦特不得不防啊，卖个圣水什么的。
心海开大语音深海的加护总感觉怪怪的，像护肤品。
怎么感觉这个场景似曾相识呢？猫猫是罢，持续注入力量。
原神刀自机有点难了，这个体量架住了。我昨天看迪娜泽黛的剧情挺意外的，过于谨慎了。刀派蒙我第一个无法接受。
这地上留的真不是火系的东西吗？久岐忍是雷系的，别一会儿再炸了
最后凯亚牺牲了自己。最后的大战发生在海上，凯亚使出全身力气冻结大海。
不可能！绝对不可能！这新三国的梗还挺多的，虽然拍的不如老三国。
"""# 剪辑方向 / 提示词
CUT_START =0                # 全局时间轴起点（秒）
CUT_END = None              # None = 到结尾
CUT_TARGET_MIN = 60 * 60    # 成片最短时长（秒）
CUT_TARGET_MAX = 70 * 60    # 成片最长时长（秒）
CUT_MIN_CLIP = 15.0         # 单段最短时长（秒）
CUT_CHUNK_SEC = 30 * 60     # 每块字幕时长：长素材用 30 分钟，避免全局挑选 prompt 超上下文
CUT_SNAP = True             # 剪辑点向外扩到静音处
CUT_DRY_RUN = False         # True = 只输出剪辑计划
CUT_WRITE_SRT = True        # 成片旁边额外写一份对齐到成片的 SRT（默认开）
CUT_MODEL = ""              # 空 = 用 edit_with_caption.py 里的默认模型
CUT_USE_PLAN = True         # 检测到现成的 <成片>.plan.json 就直接照它剪辑，不再调大模型
CUT_PLAN = ""               # 指定计划文件；空 = 按成片路径自动找 <成片名>.plan.json
CUT_EXTRA_ARGS = []         # 其余参数原样追加，例如 ["--keep-temp"]

# ---------------------------------------------------------------------------
# 运行控制
# ---------------------------------------------------------------------------
STOP_ON_ASR_FAILURE = True  # 某个分片识别失败就停下（缺字幕时剪辑端本来也跑不了）
DELETE_ASR_JSONL = False    # True = 字幕生成成功后删掉断点 .asr.jsonl


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def log(*args):
    print(*args, file=sys.stderr, flush=True)


def srt_path_for(video):
    """与视频同名的 SRT（sensevoice_srt.py 和 edit_with_caption.py 都按这个约定配对）。"""
    return os.path.splitext(video)[0] + ".srt"


def has_srt(video):
    path = srt_path_for(video)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def default_output_for(folder):
    """与 edit_with_caption.py 的默认成片路径一致：
    <文件夹>/export/<文件夹名>_edited.mp4。"""
    root = os.path.abspath(folder)
    base = os.path.basename(os.path.normpath(root))
    return os.path.join(root, EXPORT_DIR_NAME, base + "_edited.mp4")


def legacy_output_for(folder):
    """旧版默认成片路径（放在文件夹的上一级），仅用于兼容旧计划文件。"""
    root = os.path.abspath(folder)
    base = os.path.basename(os.path.normpath(root))
    return os.path.join(os.path.dirname(root), base + "_edited.mp4")


def plan_path_for(out_video):
    """成片对应的剪辑计划：<成片名>.plan.json（save_plan() 写的那个）。"""
    return os.path.splitext(out_video)[0] + ".plan.json"


def find_plan(folder, args):
    """找现成的剪辑计划：显式 --plan 优先，否则看成片旁边的 .plan.json。

    export 里没有时再找一下旧位置（文件夹的上一级），避免以前跑出来的计划
    被当成不存在而重新让大模型挑一遍。
    """
    if not args.use_plan:
        return None
    if args.plan:
        path = os.path.abspath(args.plan)
        if not os.path.isfile(path):
            raise SystemExit(f"找不到剪辑计划: {path}")
        return path
    if args.output:
        path = plan_path_for(os.path.abspath(args.output))
        return path if os.path.isfile(path) else None

    path = plan_path_for(default_output_for(folder))
    if os.path.isfile(path):
        return path
    legacy = plan_path_for(legacy_output_for(folder))
    if os.path.isfile(legacy):
        log(f"  ! 用的是旧位置的剪辑计划：{legacy}")
        log(f"    （新版计划写到 {os.path.dirname(path)}；要重新分析加 --no-plan）")
        return legacy
    return None


def run_script(cmd, title):
    """以子进程跑一个脚本（实时输出），返回 (退出码, 耗时秒)。"""
    log(f"  $ {' '.join(cmd[1:3])} ...")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    log(f"  -> {title} 退出码 {proc.returncode}，耗时 {elapsed:.1f}s")
    return proc.returncode, elapsed


def script_path(name):
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {name}，请确认它和本文件在同一目录：{path}")
    return path


# ---------------------------------------------------------------------------
# 第 1 步：逐片生成字幕
# ---------------------------------------------------------------------------
def step_asr(videos, args):
    """对每个分片调用 sensevoice_srt.py；返回失败的分片列表。"""
    log("\n=== 1/2 生成字幕（sensevoice_srt.py）===")
    log(f"设备 {args.device}，语言 {args.language}，"
        f"主播标注 {'开' if args.mark_streamer else '关'}，"
        f"字幕纠错 {'开' if args.correct else '关'}"
        f"{'，已有字幕跳过' if ASR_SKIP_EXISTING and not args.force_asr else ''}")
    if args.mark_streamer:
        if args.streamer_ref:
            log(f"主播样本 : {os.path.basename(args.streamer_ref)}"
                f"{'（前 ' + format(args.ref_sec, 'g') + 's）' if args.ref_sec > 0 else ''}")
        else:
            log("主播样本 : 未指定，自动注册")
    if args.limit_minutes > 0:
        log(f"  ! 调试模式：每个分片只识别前 {args.limit_minutes:g} 分钟，"
            f"生成的 SRT 是残缺的。正式跑之前请加 --force-asr 重跑，"
            f"或先删掉这些 .srt")

    script = script_path("sensevoice_srt.py")
    failed, done, skipped = [], 0, 0
    total_t0 = time.time()

    for i, video in enumerate(videos, 1):
        name = os.path.basename(video)
        srt = srt_path_for(video)
        if (ASR_SKIP_EXISTING and not args.force_asr
                and os.path.isfile(srt) and os.path.getsize(srt) > 0):
            skipped += 1
            log(f"[{i}/{len(videos)}] 已有字幕，跳过  {name}")
            continue

        log(f"\n[{i}/{len(videos)}] 识别  {name}")
        cmd = [sys.executable, script, video, "-o", srt,
               "--device", args.device, "--language", args.language]
        if args.limit_minutes > 0:
            cmd += ["--limit-minutes", f"{args.limit_minutes:g}"]
        if not args.correct:
            cmd += ["--no-correct"]
        if not args.mark_streamer:
            cmd += ["--no-streamer-mark"]
        else:
            if args.streamer_ref:
                cmd += ["--streamer-ref", args.streamer_ref]
            if args.streamer_threshold > 0:
                cmd += ["--streamer-threshold", f"{args.streamer_threshold:g}"]
            if args.ref_sec > 0:
                cmd += ["--ref-sec", f"{args.ref_sec:g}"]
        if ASR_MAX_HOURS > 0:
            cmd += ["--max-hours", f"{ASR_MAX_HOURS:g}"]
        cmd += list(ASR_EXTRA_ARGS)

        code, _ = run_script(cmd, "字幕")
        if code != 0:
            failed.append(video)
            log(f"  ! 识别失败: {name}")
            if STOP_ON_ASR_FAILURE:
                raise SystemExit(
                    f"分片识别失败，已停止：{name}\n"
                    f"（可修好后重跑，已有字幕会自动跳过；"
                    f"或用 --continue-on-error 跳过失败的分片）")
            continue

        done += 1
        if DELETE_ASR_JSONL:
            jsonl = os.path.splitext(srt)[0] + ".asr.jsonl"
            if os.path.isfile(jsonl):
                os.remove(jsonl)

    log(f"\n字幕汇总 : 新生成 {done} 个，跳过 {skipped} 个，失败 {len(failed)} 个，"
        f"耗时 {time.time() - total_t0:.1f}s")
    return failed


# ---------------------------------------------------------------------------
# 第 2 步：组合剪辑
# ---------------------------------------------------------------------------
def step_cut(folder, args, plan=None):
    """调用 edit_with_caption.py 做组合剪辑（给了 plan 就直接按计划渲染）。"""
    log("\n=== 2/2 自动剪辑（edit_with_caption.py）===")

    cmd = [sys.executable, script_path("edit_with_caption.py"),
           "--video-dir", folder,
           "--start", f"{args.start:g}",
           "--target-min", f"{args.target_min:g}",
           "--target-max", f"{args.target_max:g}",
           "--min-clip", f"{args.min_clip:g}",
           "--chunk-sec", f"{args.chunk_sec:g}"]
    if plan:
        log(f"  现成计划 : {plan}")
        log("  -> 直接按计划剪辑，跳过 AI 分析与挑选")
        cmd += ["--plan", plan]
    if args.end is not None:
        cmd += ["--end", f"{args.end:g}"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.use_danmaku:
        cmd += ["--use-danmaku"]
    else:
        cmd += ["--no-danmaku"]
    if not CUT_SNAP:
        cmd += ["--no-snap"]
    if args.dry_run:
        cmd += ["--dry-run"]
    if not args.write_srt:
        cmd += ["--no-output-srt"]
    if CUT_MODEL:
        cmd += ["--model", CUT_MODEL]
    cmd += ["-o", os.path.abspath(args.output) if args.output
            else default_output_for(folder)]
    cmd += list(CUT_EXTRA_ARGS)

    code, elapsed = run_script(cmd, "剪辑")
    if code != 0:
        raise SystemExit(f"剪辑失败（退出码 {code}），详见上面的输出")
    return elapsed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="文件夹内所有视频 -> SenseVoice 字幕 -> 自动组合剪辑")
    parser.add_argument("folder", nargs="?", default=VIDEO_DIR,
                        help=f"视频文件夹，默认 {VIDEO_DIR or '（未设置）'}")
    parser.add_argument("-o", "--output", default=OUTPUT_PATH,
                        help="成片路径，默认 <文件夹>/export/<文件夹名>_edited.mp4")
    parser.add_argument("--asr-only", action="store_true",
                        help="只生成字幕，不剪辑")
    parser.add_argument("--cut-only", action="store_true",
                        help="跳过字幕步骤，直接用现成 SRT 剪辑")
    parser.add_argument("--force-asr", action="store_true",
                        help="已有字幕也重新识别")
    parser.add_argument("--no-correct", dest="correct", action="store_false",
                        default=ASR_CORRECT,
                        help="不做字幕纠错（默认开，需 DEEPSEEK_API_KEY）")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="某个分片识别失败时继续处理后面的分片")
    parser.add_argument("--limit-minutes", type=float, default=ASR_LIMIT_MINUTES,
                        help="每个分片只识别前 N 分钟（调试用，0 = 全部；出来的字幕是残缺的）")
    parser.add_argument("--device", default=ASR_DEVICE,
                        help="auto / cuda:0 / cpu")
    parser.add_argument("--language", default=ASR_LANGUAGE,
                        help="zh / en / yue / ja / ko / auto")
    parser.add_argument("--streamer-ref", default=ASR_STREAMER_REF,
                        help="主播参考样本：一段主播说话的视频（mp4/flv…）或音频"
                             "（wav/mp3…）；不给则自动注册")
    parser.add_argument("--streamer-threshold", type=float,
                        default=ASR_STREAMER_THRESHOLD,
                        help="主播声纹相似度阈值，0 = 用 sensevoice_srt.py 的默认值")
    parser.add_argument("--ref-sec", type=float, default=ASR_REF_SEC,
                        help="参考样本最多用前多少秒，0 = 用默认值（300）")
    parser.add_argument("--no-streamer-mark", dest="mark_streamer",
                        action="store_false", default=ASR_MARK_STREAMER,
                        help="不标注（主播），也就不需要参考样本")
    parser.add_argument("--prompt", default=CUT_PROMPT,
                        help="剪辑方向 / 提示词，例如「眼神」")
    parser.add_argument("--start", type=float, default=CUT_START,
                        help="全局时间轴起点秒数")
    parser.add_argument("--end", type=float, default=CUT_END,
                        help="全局时间轴终点秒数，默认到结尾")
    parser.add_argument("--target-min", type=float, default=CUT_TARGET_MIN,
                        help=f"成片最短时长秒，默认 {CUT_TARGET_MIN:g}")
    parser.add_argument("--target-max", type=float, default=CUT_TARGET_MAX,
                        help=f"成片最长时长秒，默认 {CUT_TARGET_MAX:g}")
    parser.add_argument("--min-clip", type=float, default=CUT_MIN_CLIP,
                        help=f"单段最短时长秒，默认 {CUT_MIN_CLIP:g}")
    parser.add_argument("--chunk-sec", type=float, default=CUT_CHUNK_SEC,
                        help=f"每块字幕时长秒，默认 {CUT_CHUNK_SEC:g}")
    parser.add_argument("--use-danmaku", dest="use_danmaku",
                        action="store_true", default=CUT_USE_DANMAKU,
                        help="剪辑时叠加弹幕热度")
    parser.add_argument("--no-danmaku", dest="use_danmaku",
                        action="store_false", help="不使用弹幕（默认）")
    parser.add_argument("--dry-run", action="store_true", default=CUT_DRY_RUN,
                        help="只输出剪辑计划，不渲染视频")
    parser.add_argument("--no-output-srt", dest="write_srt",
                        action="store_false", default=CUT_WRITE_SRT,
                        help="不在成片旁边生成对齐到成片的 SRT（默认生成）")
    parser.add_argument("--plan", default=CUT_PLAN,
                        help="直接按已有剪辑计划（.plan.json）渲染，不再调用大模型；"
                             "默认自动找 <成片名>.plan.json")
    parser.add_argument("--no-plan", dest="use_plan", action="store_false",
                        default=CUT_USE_PLAN,
                        help="忽略现成的 .plan.json，重新让大模型挑选")
    args = parser.parse_args()

    do_asr = ASR_ENABLED and not args.cut_only
    do_cut = CUT_ENABLED and not args.asr_only
    if not do_asr and not do_cut:
        raise SystemExit("字幕和剪辑都被关掉了，没什么可做的")

    # 参考样本可能给的是相对路径，统成绝对路径（子进程的 cwd 是本目录）
    if args.streamer_ref:
        args.streamer_ref = os.path.abspath(args.streamer_ref)
        if not os.path.isfile(args.streamer_ref):
            raise SystemExit(f"找不到主播参考样本: {args.streamer_ref}")
        if not args.mark_streamer:
            log("  ! 给了 --streamer-ref 但同时关闭了主播标注，参考样本不会用到")

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"找不到视频文件夹: {folder}")
    videos = list_files(folder, VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"文件夹里没有可用的视频: {folder}")

    # 有现成剪辑计划就直按计划剪，不必再让大模型挑一遍
    plan = find_plan(folder, args) if do_cut else None
    if plan:
        plan_mtime = os.path.getmtime(plan)
        newer = [os.path.basename(v) for v in videos
                 if os.path.getmtime(v) > plan_mtime]
        if newer:
            log(f"  ! 这些分片比剪辑计划新，计划可能已过期：{', '.join(newer[:3])}"
                f"{' ...' if len(newer) > 3 else ''}（要重新分析请加 --no-plan）")

    log("=" * 66)
    log(f"文件夹   : {folder}")
    log(f"导出目录 : {os.path.join(folder, EXPORT_DIR_NAME)}")
    log(f"分片     : {len(videos)} 个（按文件名顺序 = 全局时间轴顺序）")
    for i, v in enumerate(videos, 1):
        log(f"  {i:02d} {os.path.basename(v)}")
    log(f"步骤     : {'字幕 ' if do_asr else ''}"
        f"{'+ ' if do_asr and do_cut else ''}{'剪辑' if do_cut else ''}")
    log(f"弹幕     : {'用' if args.use_danmaku else '不用'}")
    log(f"剪辑计划 : {os.path.basename(plan) if plan else '无（调大模型挑选）'}")
    log(f"成片字幕 : {'生成（对齐到成片时间轴）' if args.write_srt else '不生成'}")
    if do_asr and args.mark_streamer:
        log(f"主播样本 : {os.path.basename(args.streamer_ref) if args.streamer_ref else '自动注册'}")
    log("=" * 66)

    t_all = time.time()

    failed = step_asr(videos, args) if do_asr else []

    if do_cut:
        # 剪辑端必须有每个分片的字幕，先给出明确的缺件清单
        missing = [v for v in videos if not has_srt(v)]
        if missing:
            log("\n以下分片没有可用字幕：")
            for v in missing:
                log(f"  - {os.path.basename(v)}")
            raise SystemExit(
                "缺少字幕，无法剪辑。请先跑字幕步骤（去掉 --cut-only）。"
                f"{'（其中 ' + str(len(failed)) + ' 个是本次识别失败的）' if failed else ''}")
        step_cut(folder, args, plan)

    log(f"\n全部完成，总耗时 {time.time() - t_all:.1f}s")
    if failed:
        log(f"注意：有 {len(failed)} 个分片识别失败，本次成片是残缺时间轴上的结果")


if __name__ == "__main__":
    main()

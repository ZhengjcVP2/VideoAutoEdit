# VideoAutoEdit — Tooling Setup

Goal: take a source video + a list of `(start, end)` timestamps, cut those pieces out,
and splice them into one output video that keeps **the same (or near-identical) visual
quality and file size** as the original.

The short answer: **FFmpeg**. Everything below explains why, how to install it, and which
flags to use.

---

## 1. Recommendation at a glance

| Layer | Tool | Why |
|---|---|---|
| Video engine | **FFmpeg** (CLI) | The only practical way to cut/join video **without re-encoding** |
| Inspection | **ffprobe** (ships with FFmpeg) | Read codec params, bitrate, keyframe positions |
| Python driver | `subprocess` (stdlib) | No dependency, direct control, easiest to debug |
| Optional helper | `imageio-ffmpeg` (pip) | Bundles a static ffmpeg binary → no PATH setup needed |
| Optional syntactic sugar | `ffmpeg-python` (pip) | Declarative command building; see caveat in §4 |
| **Not** recommended | MoviePy / OpenCV VideoWriter | Both **re-encode everything**, which changes quality and size |

> The core reason for FFmpeg: it can copy compressed packets from input to output verbatim.
> Every other tool either forces a re-encode, or wraps FFmpeg anyway.

---

## 2. Install FFmpeg (Windows)

Pick **one** of the following. `winget` is the simplest.

### Option A — winget (recommended)

```powershell
winget install --id Gyan.FFmpeg -e
```

This installs a full build (with `libx264`, `libx265`, `libvpx`, `aac`, etc.) and adds it
to `PATH`.

### Option B — Chocolatey

```powershell
choco install ffmpeg -y
```

### Option C — Scoop

```powershell
scoop install ffmpeg
```

### Option D — Manual static build (no package manager)

1. Download a release zip: <https://www.gyan.dev/ffmpeg/builds/> (`ffmpeg-release-full.7z`)
   or <https://github.com/BtbN/FFmpeg-Builds/releases>.
2. Extract to e.g. `C:\Tools\ffmpeg`.
3. Add `C:\Tools\ffmpeg\bin` to **System Properties → Environment Variables → Path**.
4. Open a **new** terminal.

### Option E — pip bundle (per-project, no admin rights)

Inside the project venv:

```powershell
.\env\Scripts\Activate.ps1
pip install imageio-ffmpeg
```

Then resolve the binary from Python:

```python
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()   # e.g. ...\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-*.exe
```

Note: this bundle is a *reduced* build. It handles H.264/H.265 cut-and-copy fine, but
check available encoders with `ffmpeg -encoders` if you plan to re-encode.

### Verify

```powershell
ffmpeg -version
ffprobe -version
```

You should see a version banner and a list of `--enable-*` build options.
If PowerShell says *"ffmpeg is not recognized"*, the `PATH` change has not been picked up —
open a new terminal.

---

## 3. Why the cut is lossless (and where it isn't)

Video is stored as a sequence of **GOPs** (Groups of Pictures). Only the first frame of each
GOP is a **keyframe** (I-frame); the rest are predictive (P/B-frames) and cannot be decoded
without a preceding keyframe.

Consequences:

- **Stream copy** (`-c copy`) copies compressed packets unchanged → **identical quality,
  identical bitrate, ~zero CPU time**. But the cut *must* start on a keyframe.
- If you ask for `00:01:23.400` and the nearest earlier keyframe is at `00:01:20.000`, the
  output starts ~3.4 s early.
- To cut *exactly* at an arbitrary timestamp, the leading partial GOP must be **re-encoded**,
  which is the only place quality/size can drift.

Typical keyframe intervals: 1–2 s for streamed/encoded content, up to 10 s for camera footage.

### Strategy selection

| Strategy | Command core | Frame-accurate? | Quality vs source | Size vs source | Speed |
|---|---|---|---|---|---|
| **A. Keyframe cut (default)** | `-c copy` | No (snaps to keyframe) | **Identical** | **Identical** | Instant |
| **B. Smart cut (partial re-encode)** | hybrid | Yes | ~Identical | ~Identical | Fast |
| **C. Full re-encode, bitrate matched** | `-b:v <source>` | Yes | ~Identical | ~Identical | Slow |
| **D. Full re-encode, CRF** | `-crf 18` | Yes | Looks fine | **Usually larger** | Slow |

Recommendation: start with **A**. If the keyframe snapping causes visibly wrong cuts, move
to **B** — and pad your timestamps first (see §6).

---

## 4. Concrete commands

Let the source be `input.mp4`, and suppose a clip runs from `83.4 s` with duration `47.5 s`.

### 4.1 Probe the source first

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,bit_rate `
  -show_entries format=bit_rate,duration,size `
  -of default=noprint_wrappers=1 input.mp4
```

Record these values — you need them to confirm all segments match before concatenation.

### 4.2 Cut one segment (Strategy A — lossless)

```powershell
ffmpeg -hide_banner -y `
  -ss 83.4 -i input.mp4 -t 47.5 `
  -map 0 -c copy `
  -avoid_negative_ts make_zero `
  seg_001.mp4
```

- `-ss` **before** `-i` → fast input seek, keyframe-aligned. Correct for `-c copy`.
- `-t` (duration) is safer than `-to` here; `-to` is ambiguous once the timeline has been seeked.
- `-map 0` keeps **all** streams (extra audio tracks, subtitles) — don't omit it.
- `-avoid_negative_ts make_zero` prevents a negative start timestamp that some players and the
  concat demuxer dislike.
- Do **not** add `-movflags +faststart` to intermediates — it costs an extra rewrite and is
  only useful on the final file.

Repeat for each timestamp pair. Write the command list into `segments.txt` in order:

```text
file 'seg_001.mp4'
file 'seg_002.mp4'
file 'seg_003.mp4'
```

### 4.3 Join the segments (lossless)

```powershell
ffmpeg -hide_banner -y `
  -f concat -safe 0 -i segments.txt `
  -map 0 -c copy `
  -movflags +faststart `
  output.mp4
```

This is exact. Because every segment was copied from the same source, the codec, profile,
resolution, pixel format, timebase and SAR are already identical — the concat demuxer just
appends packets.

> Use the `concat` **demuxer** (`-f concat -i list.txt`) with `-c copy`. Do **not** use the
> `concat` **filter** (`-filter_complex "[0][1]concat=n=2"`): the filter requires decoding and
> therefore forces a re-encode of the entire video.

### 4.4 Strategy B — smart cut (frame-accurate, still ~lossless)

Only the partial GOPs at the head and tail of each clip are re-encoded; the middle is copied.
There is no single "magic flag"; the usual shape is:

1. Snap the requested start forward to the next keyframe and the requested end back to a
   keyframe → `core_start`, `core_end`.
2. Stream-copy `[core_start, core_end]` (lossless middle).
3. Encode `[requested_start, core_start)` and `[core_end, requested_end]` as short re-encoded
   snippets using the **same encoder family and settings as the source**.
4. Concat all three.

This needs identical encoder parameters across snippet and middle, otherwise the concat
demuxer will produce a broken stream. In practice you either (a) accept Strategy A, or
(b) go to Strategy C which is much simpler to reason about.

**GUI prior art worth studying:** *LosslessCut* implements exactly this workflow and can
inspect its output to confirm the ffmpeg invocations.

### 4.5 Strategy C — full re-encode with matched bitrate

Use only when frame accuracy is mandatory. Match the source's **bitrate and encoder family**,
not a CRF constant — re-encoding a low-bitrate source at `-crf 18` typically produces a
*larger* file than the original.

```powershell
ffmpeg -hide_banner -y `
  -ss 83.4 -t 47.5 -i input.mp4 `
  -map 0 `
  -c:v libx264 -preset slow -b:v 4500k -maxrate 5500k -bufsize 9000k `
  -c:a aac -b:a 192k `
  seg_001.mp4
```

`4500k` above = the `bit_rate` you read from `ffprobe` in §4.1. Keep `-preset slow` (better
quality per bit), and pick `libx264` or `libx265` to match the source codec.

---

## 5. Matching the source exactly

Before concatenating, confirm every segment reports the same values:

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=codec_name,profile,width,height,pix_fmt,time_base,sample_aspect_ratio `
  -of csv=p=0 seg_001.mp4
```

A mismatch in any of `codec_name`, `profile`, `width`, `height`, `pix_fmt`, `time_base` or
`sample_aspect_ratio` will break `-c copy` concatenation. If the source is **HDR**, also check
`color_primaries`, `color_transfer`, `color_space`, `color_range` — stream copy preserves
them, re-encode does not unless you pass them through explicitly.

---

## 6. Practical checklist

- [ ] **Pad your cuts.** If timestamps come from external analysis, add ~0.3–0.5 s of head/tail
      slack. It hides keyframe snapping and usually *improves* the edit.
- [ ] **Snap to real keyframes** instead of guessing. List them once:
      ```powershell
      ffprobe -v error -select_streams v:0 -skip_frame nokey `
        -show_entries frame=pts_time -of csv=p=0 input.mp4
      ```
      Then map each requested `start` to the nearest listed `pts_time` `<= start`.
- [ ] **Keep the source untouched.** Never write segments over `input.mp4`.
- [ ] **Use absolute paths** or a dedicated working folder for `segments.txt` (relative paths
      resolve against the *list file's* directory).
- [ ] **Quote paths with spaces** in `segments.txt`, and use `-safe 0` when paths are absolute.
- [ ] **Long sources:** feed segments to the concat list incrementally; you don't need all
      clips in memory.
- [ ] **Subtitle/audio offset:** stream copy keeps timestamps per-stream. If audio drifts after
      concat, re-run with `-async 1` only on re-encoded tracks.

---

## 7. Python sketch (no extra dependencies)

```python
import subprocess

def cut(src: str, start: float, dur: float, dst: str) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-map", "0", "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        dst,
    ], check=True)

def concat(segments: list[str], dst: str) -> None:
    with open("segments.txt", "w", encoding="utf-8") as f:
        for p in segments:
            f.write(f"file '{p}'\n")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", "segments.txt",
        "-map", "0", "-c", "copy", "-movflags", "+faststart",
        dst,
    ], check=True)
```

`check=True` turns a non-zero exit code into an exception — always inspect stderr on failure.

---

## 8. Optional additions

| Need | Package / tool |
|---|---|
| Loudness normalization across clips | `ffmpeg -af loudnorm` or `ffmpeg-normalize` (pip) |
| Burning subtitles from your timestamp list | `ffmpeg -vf subtitles=out.srt` (forces video re-encode) |
| Silence/energy-based auto cutting (prior art) | `auto-editor` (pip), wraps FFmpeg |
| Sanity-check output duration | `ffprobe -show_entries format=duration` |
| Download Bilibili source video + danmaku | **BBDown** (bundled, see §10) |

---

## 9. Install summary

```powershell
# 1. Engine
winget install --id Gyan.FFmpeg -e

# 2. Optional Python helpers (inside the project venv)
.\env\Scripts\Activate.ps1
pip install imageio-ffmpeg

# 3. Verify
ffmpeg -version
ffprobe -version
```

Nothing else is required. FFmpeg already provides the cutting, joining, probing and
re-encoding needed for this project — no additional video library has to be installed.

---

## 10. Downloading source material from Bilibili (BBDown)

BBDown is already installed **inside this project** — no admin rights, no PATH changes:

```
tools\BBDown\BBDown.exe      # v1.6.3, self-contained win-x64 (~18 MB)
```

### Commands

```powershell
# 1) 只解析、不下载：列出该视频所有可用清晰度
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" -info

# 2) 下载 1080P60（名字要跟 -info 列表里的完全一致）
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" -q "1080P60 高帧率,1080P 高码率,1080P 高清"

# 3) 顺便保存弹幕 XML，直接交给 edit_with_caption.py
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" -q "1080P60 高帧率" -dd

# 4) 清晰度名字对不上时，用交互式选择
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" -ia

# 5) 只要弹幕 / 只要字幕 / 只要封面
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" --danmaku-only

# 6) 合集或分P：3-5 是范围，也可以用 ALL / LAST
.\tools\BBDown\BBDown.exe "BV1xxxxxxxxx" -p 3-5
```

### Notes

- **登录**：首次下载高清需要扫码登录 —— `tools\BBDown\BBDown.exe login`，凭据写在
  `BBDown.data`（exe 同目录，勿提交到仓库）。
- **权限**：未登录只给 360P/480P；1080P 需登录；1080P60、1080P 高码率、4K、HDR、
  杜比视界属大会员档（账号没有会员时 `-info` 里就不会出现这些流）。
- **混流**：BBDown 下载的是 DASH 分离流（video `.m4s` + audio `.m4s`），结束后会自动
  调用 `ffmpeg` 合成 mp4，所以 **ffmpeg 必须在 PATH**（本项目已安装）。
- **清晰度名**来自 `-info` 输出的方括号内容（如 `[1080P 高清]`、`[1080P60 高帧率]`），
  `-q` 按逗号分隔的顺序做优先级匹配，全部不匹配时回退到最高可用。
- **编码**：`-e "avc,hevc"` 可指定优先 AVC（H.264），兼容性更好、剪辑更快。

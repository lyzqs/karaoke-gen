#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from karaoke_gen.local_jp_renderer import render_local_jp_video


DEFAULT_AUDIO = Path("/root/clawteam-lab/samples/nicokara-jp/sample.mp3")
DEFAULT_BACKGROUND = Path("/root/clawteam-lab/samples/nicokara-jp/background.png")
DEFAULT_LRC = Path("/root/clawteam-lab/samples/nicokara-jp/sample.lrc")
DEFAULT_TXT = Path("/root/clawteam-lab/samples/nicokara-jp/lyrics.txt")
DEFAULT_OUTPUT_DIR = Path("tmp/local-jp-sample")


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "local_jp_sample"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the nicokara JP sample to a Telegram-friendly 1080p MP4.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help=f"Audio file. Default: {DEFAULT_AUDIO}")
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND, help=f"Background image/video. Default: {DEFAULT_BACKGROUND}")
    parser.add_argument("--timing-lrc", type=Path, default=DEFAULT_LRC, help=f"Timed LRC file or TXT timing bridge. Default: {DEFAULT_LRC}")
    parser.add_argument("--lyrics-txt", type=Path, help=f"Optional plain TXT lyrics. If set, --timing-lrc is used as the alignment bridge. Example default: {DEFAULT_TXT}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--title", default="Mutual panorama local-jp 1080p", help="Output stem/title for generated files.")
    parser.add_argument("--resolution", choices=["1080p"], default="1080p", help="Target delivery resolution.")
    args = parser.parse_args()

    result = render_local_jp_video(
        audio_path=args.audio,
        background_path=args.background,
        output_dir=args.output_dir,
        output_stem=_safe_stem(args.title),
        timing_lrc_path=None if args.lyrics_txt else args.timing_lrc,
        lyrics_txt_path=args.lyrics_txt,
        txt_timing_bridge_lrc_path=args.timing_lrc if args.lyrics_txt else None,
        resolution=args.resolution,
    )

    print(f"MP4: {result.video_path}")
    print(f"ASS: {result.ass_path}")
    print(f"Timeline: {result.timeline_path}")
    print(f"TXT: {result.txt_path}")
    print(f"LRC: {result.lrc_path}")
    print(f"Duration ms: {result.duration_ms}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Manual frame extraction script (for debugging).
pipeline.py extracts frames automatically at run time; this script is for standalone debugging.

Usage:
    python scripts/extract_frames.py --task maze --split eval --difficulty hard
    python scripts/extract_frames.py --task maze --split eval --difficulty hard --max_samples 5
"""

import csv
import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "frames_cache"


def resolve_path(csv_path: str) -> Path:
    parts = csv_path.split("/")
    if parts[0] in ("eval", "train"):
        parts = parts[1:]
    return RAW_DATA_DIR / Path(*parts)


def extract(video_path: Path, out_dir: Path, fps: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("frame_*.jpg")):
        print(f"  [skip] already extracted: {out_dir.name}")
        return
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%03d.jpg"),
        "-y", "-loglevel", "quiet"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed: {result.stderr.decode()[:200]}")
    else:
        n = len(list(out_dir.glob("frame_*.jpg")))
        print(f"  [ok] {n} frames → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       required=True)
    parser.add_argument("--split",      default="eval")
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--fps",        type=float, default=2.0)
    args = parser.parse_args()

    meta_base = RAW_DATA_DIR / "metadata" / args.split
    count = 0

    for folder in sorted(meta_base.iterdir()):
        parts = folder.name.rsplit("_", 2)
        if len(parts) != 3:
            continue
        t, variant, difficulty = parts
        if t != args.task or difficulty != args.difficulty:
            continue

        csv_path = folder / "metadata.csv"
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        print(f"\n[{folder.name}] {len(rows)} samples")
        for row in rows:
            vid_path = resolve_path(row["video"])
            vid_stem = vid_path.stem
            sample_id = f"{args.task}_{variant}_{vid_stem}"
            out_dir = OUTPUT_DIR / args.task / variant / sample_id
            extract(vid_path, out_dir, fps=args.fps)
            count += 1
            if args.max_samples > 0 and count >= args.max_samples:
                print(f"\nReached max_samples={args.max_samples}, stopping.")
                return

    print(f"\nDone. Extracted {count} videos.")


if __name__ == "__main__":
    main()

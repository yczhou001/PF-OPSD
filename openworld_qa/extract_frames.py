#!/usr/bin/env python3
"""
OpenWorldQA Frame Extraction Script
Extract frames from videos at a fixed ~0.5s interval, capped at the first 10s.

Strategy:
  - Cover the first 10 seconds of the video (or full video if shorter)
  - Extract one frame every ~0.5s → 4 to 12 frames depending on video length
  - No fixed anchor: all frames are named frame_001 ~ frame_N
  - SceneAnalyst agent will decide which frame is the anchor at pipeline time

Usage:
    python extract_frames.py --video <path_to_video.mp4>
    python extract_frames.py --video_dir <directory_with_videos>

Output: output/frames/{video_id}/frame_001.jpg, frame_002.jpg, ..., frame_N.jpg
        output/frames/{video_id}/metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── Extraction parameters ──────────────────────────────────────────────────────

# Only look at the first N seconds of each video.
# Charades actions typically complete within 3-8s; 10s is sufficient and avoids
# unrelated actions later in the clip.
EFFECTIVE_WINDOW_SEC = 10.0
DEFAULT_WINDOW_SEC   = EFFECTIVE_WINDOW_SEC   # alias for sample_and_extract.py

# Target interval between frames (seconds).
# 0.5s captures fast actions (throw, pour, collide) with 2-3 intermediate frames.
FRAME_INTERVAL_SEC = 0.5

# Hard cap on total frames per video to control API token cost.
MAX_FRAMES = 12

# Minimum frames to extract regardless of video length.
MIN_FRAMES = 4


# ──────────────────────────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe. Returns 0.0 on error."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error reading duration for {video_path}: {result.stderr}", file=sys.stderr)
        return 0.0
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def compute_timestamps(duration: float, window_sec: float | None = None,
                       num_frames: int | None = None) -> list[float]:
    """
    Compute frame timestamps for a video of the given duration.

    Args:
        duration:   Total video duration in seconds.
        window_sec: Only use the first this many seconds (default: EFFECTIVE_WINDOW_SEC).
        num_frames: Number of frames to extract (default: derived from FRAME_INTERVAL_SEC).

    Returns a list of N timestamps in [0, min(duration, window_sec)].
    """
    window = min(duration, window_sec if window_sec is not None else EFFECTIVE_WINDOW_SEC)

    if num_frames is not None:
        n = max(1, num_frames)
    else:
        n = max(MIN_FRAMES, min(int(window / FRAME_INTERVAL_SEC), MAX_FRAMES))

    if n == 1:
        return [0.0]

    # Evenly spaced from t=0 to t=window (inclusive of both endpoints)
    step = window / (n - 1)
    timestamps = [round(i * step, 3) for i in range(n)]

    # Safety: clamp to valid range
    timestamps = [min(max(ts, 0.0), duration * 0.999) for ts in timestamps]
    return timestamps


def extract_frames(video_path: str, output_dir: str,
                   num_frames: int | None = None,
                   window_sec: float | None = None) -> dict:
    """
    Extract frames from a single video and write them + metadata to output_dir.

    Args:
        video_path:  Path to the source video file.
        output_dir:  Root directory; frames go into output_dir/{video_stem}/.
        num_frames:  How many frames to extract (default: derived from FRAME_INTERVAL_SEC).
        window_sec:  Only use the first this many seconds (default: EFFECTIVE_WINDOW_SEC).

    Returns a metadata dict describing the extracted frames.
    """
    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem

    frames_subdir = os.path.join(output_dir, video_name)
    os.makedirs(frames_subdir, exist_ok=True)

    duration = get_video_duration(video_path)
    if duration == 0:
        return {"status": "error", "message": f"Could not read video duration: {video_path}"}

    timestamps = compute_timestamps(duration, window_sec=window_sec, num_frames=num_frames)

    extracted = []
    for idx, ts in enumerate(timestamps):
        filename = f"frame_{idx + 1:03d}.jpg"
        output_path = os.path.join(frames_subdir, filename)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: failed to extract frame at {ts:.3f}s: {result.stderr.strip()}",
                  file=sys.stderr)
            continue

        extracted.append({
            "filename": filename,
            "path": os.path.relpath(output_path, os.getcwd()),
            "timestamp_sec": ts,
            "size_bytes": os.path.getsize(output_path),
        })

    effective_window = min(duration, window_sec if window_sec is not None else EFFECTIVE_WINDOW_SEC)
    actual_interval = (effective_window / (len(extracted) - 1)) if len(extracted) > 1 else 0.0

    metadata = {
        "video_path": video_path,
        "video_name": video_name,
        "duration_sec": round(duration, 2),
        "effective_window_sec": round(effective_window, 2),
        "frame_interval_sec": round(FRAME_INTERVAL_SEC, 2),
        "actual_interval_sec": round(actual_interval, 3),
        "num_frames": len(extracted),
        "frames": extracted,
        # anchor_frame is intentionally absent here — SceneAnalyst selects it.
        "status": "ok",
    }

    meta_path = os.path.join(frames_subdir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="OpenWorldQA Frame Extraction — fixed 0.5s interval, first 10s of video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_frames.py --video raw_data/charades/Charades_v1_480/BCQWJ.mp4
  python extract_frames.py --video_dir raw_data/charades/Charades_v1_480/
  python extract_frames.py --video_list my_videos.txt

Extraction strategy:
  - Covers the first 10s of each video (full video if shorter).
  - One frame every ~0.5s → 4 to 12 frames per video.
  - All frames named frame_001.jpg ... frame_N.jpg (no anchor pre-set).
  - SceneAnalyst agent selects the anchor frame during pipeline execution.
        """
    )
    parser.add_argument("--video",      type=str, nargs="+", help="Path(s) to video file(s)")
    parser.add_argument("--video_dir",  type=str, help="Directory containing .mp4 files")
    parser.add_argument("--video_list", type=str, help="Text file listing one video path per line")
    parser.add_argument("--output_dir", type=str, default="output/frames",
                        help="Output directory for extracted frames (default: output/frames)")
    parser.add_argument("--ext", type=str, default=".mp4",
                        help="Video extension filter for --video_dir (default: .mp4)")

    args = parser.parse_args()

    if not any([args.video, args.video_dir, args.video_list]):
        parser.error("One of --video, --video_dir, or --video_list is required")

    videos: list[str] = []
    if args.video:
        videos.extend(args.video)
    if args.video_dir:
        for f in sorted(os.listdir(args.video_dir)):
            if f.lower().endswith(args.ext):
                videos.append(os.path.join(args.video_dir, f))
    if args.video_list:
        with open(args.video_list, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    videos.append(line)

    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    print(f"OpenWorldQA Frame Extraction")
    print(f"  Videos   : {len(videos)}")
    print(f"  Strategy : first {EFFECTIVE_WINDOW_SEC}s, ~{FRAME_INTERVAL_SEC}s interval, "
          f"{MIN_FRAMES}-{MAX_FRAMES} frames per video")
    print(f"  Output   : {args.output_dir}")
    print("=" * 60)

    results = []
    success_count = 0
    fail_count = 0

    for i, video in enumerate(videos, 1):
        name = Path(video).name
        print(f"\n[{i}/{len(videos)}] {name}")

        result = extract_frames(video, args.output_dir)
        results.append(result)

        if result["status"] == "ok":
            success_count += 1
            print(f"  OK — {result['num_frames']} frames "
                  f"(interval={result['actual_interval_sec']:.2f}s, "
                  f"window={result['effective_window_sec']:.1f}s) "
                  f"→ {result['frames'][0]['path'].rsplit('/', 1)[0]}/")
        else:
            fail_count += 1
            print(f"  FAIL — {result.get('message', 'Unknown error')}")

    print("\n" + "=" * 60)
    print(f"Done: {success_count} succeeded, {fail_count} failed, {len(videos)} total")

    summary = {
        "total_videos": len(videos),
        "success": success_count,
        "failed": fail_count,
        "effective_window_sec": EFFECTIVE_WINDOW_SEC,
        "frame_interval_sec": FRAME_INTERVAL_SEC,
        "max_frames": MAX_FRAMES,
        "output_dir": args.output_dir,
        "results": results,
    }
    summary_path = os.path.join(args.output_dir, "extraction_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

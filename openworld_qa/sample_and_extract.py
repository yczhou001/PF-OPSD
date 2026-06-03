#!/usr/bin/env python3
"""
OpenWorldQA Dataset Sampling + Frame Extraction

Randomly samples videos from 4 datasets according to specified quotas,
then extracts frames using the event-window strategy.

Sampling plan (default 5000 total):
  - SomethingV2  : 2200  (44%) → C1/C2/C3/C5/C6/C8/C9
  - Charades     : 1200  (24%) → C7/C9/C11/C12
  - Oops         : 900   (18%) → C4/C5/C6/C10/C12
  - CharadesEgo  : 700   (14%) → C2/C9/C11

Usage:
    python sample_and_extract.py
    python sample_and_extract.py --total 5000 --num_frames 4 --window_sec 4.0
    python sample_and_extract.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# extract_frames.py is in the same package directory
from extract_frames import extract_frames, DEFAULT_WINDOW_SEC

# =============================================================================
# Dataset registry
# =============================================================================
# Raw video data root.  Set RAW_DATA_ROOT in your environment to point to your
# downloaded datasets, or place them under openworld_qa/raw_data/ (default).
BASE = Path(os.environ.get("RAW_DATA_ROOT", str(Path(__file__).parent / "raw_data")))

DATASETS = {
    "sthv2": {
        "root": BASE / "sthv2/extracted/20bn-something-something-v2",
        "ext": ".webm",
        "quota": 2200,
        "desc": "SomethingV2 (hand-object interaction)",
    },
    "charades": {
        "root": BASE / "charades/Charades_v1_480/Charades_v1_480",
        "ext": ".mp4",
        "quota": 1200,
        "desc": "Charades (indoor activities, 3rd person)",
    },
    "oops": {
        "root": BASE / "oops/oops_dataset/oops_video/train",
        "ext": ".mp4",
        "quota": 900,
        "desc": "Oops (accidents & fails, physical events)",
    },
    "charades_ego": {
        "root": BASE / "charades_ego/CharadesEgo_v1_480",
        "ext": ".mp4",
        "quota": 700,
        "desc": "CharadesEgo (indoor activities, 1st person)",
    },
}
# =============================================================================


def collect_videos(dataset_cfg: dict, quota: int, rng: random.Random) -> list[str]:
    """Return a random sample of up to `quota` video paths from this dataset."""
    root = dataset_cfg["root"]
    ext = dataset_cfg["ext"]
    all_videos = sorted(root.glob(f"*{ext}"))
    if not all_videos:
        print(f"  [Warning] No {ext} files found in {root}")
        return []
    sampled = rng.sample(all_videos, min(quota, len(all_videos)))
    return [str(p) for p in sampled]


def extract_one(args_tuple) -> dict:
    """Worker function: extract frames for one video."""
    video_path, output_dir, num_frames, window_sec, dataset_name = args_tuple
    result = extract_frames(video_path, output_dir, num_frames, window_sec)
    result["dataset"] = dataset_name
    return result


def main():
    parser = argparse.ArgumentParser(description="OpenWorldQA Sampling + Frame Extraction")
    parser.add_argument("--total",       type=int,   default=5000,               help="Total videos to sample (default: 5000)")
    parser.add_argument("--output_dir",  type=str,   default="output/frames",    help="Output directory for frames")
    parser.add_argument("--num_frames",  type=int,   default=4,                  help="Frames per video (default: 4)")
    parser.add_argument("--window_sec",  type=float, default=DEFAULT_WINDOW_SEC, help=f"Event window in seconds (default: {DEFAULT_WINDOW_SEC}s)")
    parser.add_argument("--num_workers", type=int,   default=16,                 help="Parallel ffmpeg workers (default: 16)")
    parser.add_argument("--seed",        type=int,   default=42,                 help="Random seed (default: 42)")
    parser.add_argument("--dry_run",     action="store_true",                    help="Show plan without extracting")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Scale quotas proportionally if --total differs from 5000
    scale = args.total / 5000.0
    datasets_to_use = {
        name: {**cfg, "quota": max(1, round(cfg["quota"] * scale))}
        for name, cfg in DATASETS.items()
    }

    print("=" * 60)
    print("OpenWorldQA Sample & Extract")
    print("=" * 60)
    print(f"Total target:  {args.total}")
    print(f"Num frames:    {args.num_frames} (window: {args.window_sec}s)")
    print(f"Workers:       {args.num_workers}")
    print(f"Seed:          {args.seed}")
    print()
    print("Sampling plan:")
    for name, cfg in datasets_to_use.items():
        print(f"  {name:15s}: {cfg['quota']:5d}  — {cfg['desc']}")
    print("=" * 60)

    # --- Collect video paths ---
    all_tasks = []
    for name, cfg in datasets_to_use.items():
        videos = collect_videos(cfg, cfg["quota"], rng)
        print(f"  {name}: {len(videos)} videos sampled (available: {len(list(cfg['root'].glob('*' + cfg['ext'])))})")
        for v in videos:
            all_tasks.append((v, args.output_dir, args.num_frames, args.window_sec, name))

    print(f"\nTotal tasks: {len(all_tasks)}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without extraction.")
        return

    # --- Extract frames in parallel ---
    os.makedirs(args.output_dir, exist_ok=True)
    success, failed = 0, 0
    dataset_counts = {name: 0 for name in datasets_to_use}

    print(f"\nStarting extraction with {args.num_workers} workers...\n{'=' * 60}")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(extract_one, task): task for task in all_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            ds = result.get("dataset", "unknown")
            if result["status"] == "ok":
                success += 1
                dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
            else:
                failed += 1
                print(f"  FAIL [{ds}] {result.get('message', '')}")

            if i % 200 == 0 or i == len(all_tasks):
                pct = i / len(all_tasks) * 100
                print(f"  [{i:5d}/{len(all_tasks)}] {pct:.1f}%  ✓{success}  ✗{failed}")
                sys.stdout.flush()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Succeeded: {success}")
    print(f"  Failed:    {failed}")
    print()
    print("Per-dataset breakdown:")
    for name, cnt in dataset_counts.items():
        print(f"  {name:15s}: {cnt}")

    # Save sampling manifest
    manifest_path = Path(args.output_dir) / "sampling_manifest.json"
    manifest = {
        "total_requested": args.total,
        "total_extracted": success,
        "failed": failed,
        "num_frames": args.num_frames,
        "window_sec": args.window_sec,
        "seed": args.seed,
        "dataset_counts": dataset_counts,
        "datasets": {
            name: {"quota": cfg["quota"], "root": str(cfg["root"]), "ext": cfg["ext"]}
            for name, cfg in datasets_to_use.items()
        }
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest saved: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate any VLM on the OpenWorldQA test set.

Each sample: anchor frame image + 4-choice MCQ → predict A/B/C/D.
The anchor frame path is resolved from the `video_path` field stored in
each sample JSON (set during dataset creation by pipeline.py).

Usage:
    python evaluate.py                                    # default: test split, gpt-5.4
    python evaluate.py --split test                       # evaluate on test set
    python evaluate.py --split train                      # evaluate on train set
    python evaluate.py --model your-model-name            # specify model
    python evaluate.py --frames_dir /path/to/frames       # custom frames dir
    python evaluate.py --num_workers 8

Environment variables:
    OPENAI_API_KEY   — required
    OPENAI_API_BASE  — optional (default: https://api.openai.com/v1)
"""

import argparse
import base64
import json
import os
import re
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY      = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

DEFAULT_MODEL = "gpt-5.4"
BASE_DIR      = Path(__file__).parent.parent          # repo root

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(sample: dict) -> str:
    opts = "\n".join(f"  {k}: {v}" for k, v in sample["options"].items())
    return (
        "Look at the image carefully and answer the following multiple-choice question "
        "about physical world understanding.\n\n"
        f"Question: {sample['question']}\n\n"
        f"Options:\n{opts}\n\n"
        'Reply with a JSON object containing only the key "answer" with value A, B, C, or D. '
        'Example: {"answer": "A"}'
    )


def extract_answer(text: str) -> str:
    m = re.search(r'\{\s*"answer"\s*:\s*"([ABCD])"\s*\}', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b([ABCD])\b', text.strip())
    if m:
        return m.group(1).upper()
    return ""


def resolve_anchor(sample: dict, frames_dir_override: Path | None) -> Path | None:
    """
    Resolve the anchor frame path.

    Priority:
      1. frames_dir_override / {video_id} / {anchor_frame}
      2. sample["video_path"] / {anchor_frame}   (absolute or relative to repo root)
    """
    anchor_name = sample.get("anchor_frame", "frame_001")
    if not anchor_name.endswith(".jpg"):
        anchor_name += ".jpg"

    if frames_dir_override is not None:
        video_id = Path(sample["video_path"]).name
        p = frames_dir_override / video_id / anchor_name
        if p.exists():
            return p

    # Try video_path directly
    vp = Path(sample["video_path"])
    if not vp.is_absolute():
        vp = BASE_DIR / vp
    p = vp / anchor_name
    if p.exists():
        return p

    return None


def eval_one(sample: dict, model: str,
             frames_dir_override: Path | None,
             max_retries: int = 3) -> dict:
    sid = sample["sample_id"]
    anchor_path = resolve_anchor(sample, frames_dir_override)

    if anchor_path is None:
        return {
            "sample_id": sid,
            "category": sample["category"],
            "difficulty": sample.get("difficulty", "Hard"),
            "answer": sample["answer"],
            "predicted": "",
            "correct": False,
            "error": f"anchor frame not found for {sid}",
        }

    b64    = encode_image(anchor_path)
    prompt = build_prompt(sample)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
            )
            raw       = resp.choices[0].message.content
            predicted = extract_answer(raw)
            return {
                "sample_id":    sid,
                "category":     sample["category"],
                "difficulty":   sample.get("difficulty", "Hard"),
                "answer":       sample["answer"],
                "predicted":    predicted,
                "correct":      predicted == sample["answer"],
                "raw_response": raw,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return {
                    "sample_id":  sid,
                    "category":   sample["category"],
                    "difficulty": sample.get("difficulty", "Hard"),
                    "answer":     sample["answer"],
                    "predicted":  "",
                    "correct":    False,
                    "error":      str(e),
                }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM on the OpenWorldQA benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split",       default="test", choices=["test", "train"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Model name to evaluate.")
    parser.add_argument("--split_dir",   default="",
                        help="Path to OpenWorldQA_split directory (default: <repo>/OpenWorldQA_split).")
    parser.add_argument("--frames_dir",  default="",
                        help="Base directory for extracted frames "
                             "(default: resolved from sample's video_path field).")
    parser.add_argument("--output",      default="",
                        help="Output JSON file (default: eval_results/<model>_<split>.json).")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # Resolve paths
    split_dir = (
        Path(args.split_dir) if args.split_dir
        else BASE_DIR / "OpenWorldQA_split" / args.split
    )
    frames_dir_override = Path(args.frames_dir) if args.frames_dir else None

    out_dir  = BASE_DIR / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    out_file   = (
        Path(args.output) if args.output
        else out_dir / f"{model_slug}_{args.split}.json"
    )

    # Load samples
    samples = [json.loads(f.read_text()) for f in sorted(split_dir.glob("*.json"))]
    print(f"Model    : {args.model}")
    print(f"Split    : {args.split}  ({len(samples)} samples)")
    print(f"Workers  : {args.num_workers}")
    print(f"Output   : {out_file}")
    print()

    results     = []
    lock        = threading.Lock()
    done_count  = [0]
    error_count = [0]

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(eval_one, s, args.model, frames_dir_override): s["sample_id"]
            for s in samples
        }
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                done_count[0] += 1
                if "error" in r:
                    error_count[0] += 1
                if done_count[0] % 50 == 0 or done_count[0] == len(samples):
                    correct_n = sum(1 for x in results if x["correct"])
                    print(f"  [{done_count[0]:>4}/{len(samples)}]  "
                          f"acc={correct_n/done_count[0]:.4f}  "
                          f"errors={error_count[0]}")

    # Aggregate
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])

    cat_stats  = defaultdict(lambda: {"correct": 0, "total": 0})
    diff_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_stats[r["category"]]["total"] += 1
        diff_stats[r["difficulty"]]["total"] += 1
        if r["correct"]:
            cat_stats[r["category"]]["correct"] += 1
            diff_stats[r["difficulty"]]["correct"] += 1

    output_data = {
        "model":            args.model,
        "split":            args.split,
        "total_samples":    total,
        "overall_accuracy": correct / total if total else 0,
        "category_accuracy": {
            cat: {"accuracy": v["correct"] / v["total"],
                  "correct": v["correct"], "total": v["total"]}
            for cat, v in sorted(cat_stats.items())
        },
        "difficulty_accuracy": {
            diff: {"accuracy": v["correct"] / v["total"],
                   "correct": v["correct"], "total": v["total"]}
            for diff, v in sorted(diff_stats.items())
        },
        "per_sample_results": sorted(results, key=lambda x: x["sample_id"]),
    }

    out_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))

    # Print summary
    oa = output_data["overall_accuracy"]
    print(f"\n{'='*55}")
    print(f"  Model   : {args.model}")
    print(f"  Split   : {args.split}  ({total} samples)")
    print(f"  Overall : {oa:.4f}  ({correct}/{total})")
    print(f"{'='*55}")
    print(f"  {'Category':<25} {'Acc':>7}  {'Correct':>8}  {'Total':>6}")
    print(f"  {'-'*50}")
    for cat, v in output_data["category_accuracy"].items():
        print(f"  {cat:<25} {v['accuracy']:>7.4f}  {v['correct']:>8}  {v['total']:>6}")
    print(f"\n  Difficulty breakdown:")
    for diff, v in output_data["difficulty_accuracy"].items():
        print(f"    {diff:<15} {v['accuracy']:.4f}  ({v['correct']}/{v['total']})")
    print(f"\n  Errors: {error_count[0]}")
    print(f"  Saved : {out_file}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

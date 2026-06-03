#!/usr/bin/env python3
"""
Re-shuffle the option order (A/B/C/D) for every question in output/reviewed/,
and re-determine the answer field using correct_value.

Original files are overwritten in place (backed up to output/reviewed_backup/ first).

Usage:
    python scripts/shuffle_options.py
    python scripts/shuffle_options.py --seed 42
"""

import json
import random
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REVIEWED_DIR = PROJECT_ROOT / "output" / "reviewed"
BACKUP_DIR   = PROJECT_ROOT / "output" / "reviewed_backup"

OPTION_KEYS = ["A", "B", "C", "D"]


def shuffle_one(qa: dict, rng: random.Random) -> dict:
    options = qa["options"]          # {"A": "10", "B": "11", ...}
    correct_value = qa["correct_value"]  # the value of the correct answer, e.g. "10" or "west"

    # Shuffle the value list and reassign to A/B/C/D
    values = list(options.values())
    rng.shuffle(values)

    new_options = {k: v for k, v in zip(OPTION_KEYS, values)}

    # Find the new key that corresponds to correct_value
    new_answer = next(k for k, v in new_options.items() if v == correct_value)

    qa = dict(qa)
    qa["options"] = new_options
    qa["answer"]  = new_answer
    return qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025,
                        help="random seed (default 2025, ensures reproducibility)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    qa_files = sorted(REVIEWED_DIR.glob("VRB_*.json"))
    if not qa_files:
        print(f"[ERROR] No VRB_*.json files found in {REVIEWED_DIR}")
        return

    # Back up original files
    if BACKUP_DIR.exists():
        print(f"[Info] Backup already exists at {BACKUP_DIR}, skipping backup.")
    else:
        print(f"[Backup] Copying {len(qa_files)} files -> {BACKUP_DIR} ...")
        shutil.copytree(REVIEWED_DIR, BACKUP_DIR)
        print(f"[Backup] Done.")

    # Count answer distribution before and after shuffle
    before_counter = {"A": 0, "B": 0, "C": 0, "D": 0}
    after_counter  = {"A": 0, "B": 0, "C": 0, "D": 0}

    for f in qa_files:
        qa = json.loads(f.read_text(encoding="utf-8"))
        before_counter[qa["answer"]] += 1

        qa_shuffled = shuffle_one(qa, rng)
        after_counter[qa_shuffled["answer"]] += 1

        f.write_text(json.dumps(qa_shuffled, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(qa_files)
    print(f"\nShuffled {total} files  (seed={args.seed})")
    print(f"\n{'Option':>6}  {'before':>10}  {'after':>10}")
    print("-" * 32)
    for k in OPTION_KEYS:
        b = before_counter[k]
        a = after_counter[k]
        print(f"  {k}   {b:>5} ({b/total*100:4.1f}%)   {a:>5} ({a/total*100:4.1f}%)")
    print(f"\n[Done] Output: {REVIEWED_DIR}")
    print(f"[Done] Backup: {BACKUP_DIR}")


if __name__ == "__main__":
    main()

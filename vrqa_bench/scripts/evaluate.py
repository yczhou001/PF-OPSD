#!/usr/bin/env python3
"""
VRBench VQA evaluation script

Reads QA from output/reviewed/, runs inference with the given model (input_image only),
and selects A/B/C/D. Results are written to evals/results/<model>.json, logs to evals/logs/<model>.log.

Usage:
    python scripts/evaluate.py --model gpt-5.4
    python scripts/evaluate.py --model gemini-3-flash-preview
    python scripts/evaluate.py --model gpt-5.4 --num_workers 8
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY      = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

PROJECT_ROOT = Path(__file__).parent.parent
REVIEWED_DIR = PROJECT_ROOT / "output" / "reviewed"
EVAL_DIR     = PROJECT_ROOT / "evals" / "results"
LOG_DIR      = PROJECT_ROOT / "evals" / "logs"

PROMPT = (
    "You are given a spatial puzzle image showing the initial state of a maze or puzzle.\n"
    "Answer the following multiple-choice question by selecting the single best option.\n"
    "Reply with JSON only, exactly in this format: {\"answer\": \"A\"}\n"
    "The answer must be exactly one of: A, B, C, or D. No explanation needed."
)


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(model: str, question: str, options: dict, image_path: str,
               max_retries: int = 3) -> str | None:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    options_text = "\n".join(f"  {k}: {v}" for k, v in options.items())
    text_prompt = f"{PROMPT}\n\nQUESTION: {question}\n\nOPTIONS:\n{options_text}"

    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    content = [
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            raw = resp.choices[0].message.content
            # parse JSON answer
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                ans = parsed.get("answer", "").strip().upper()
                if ans in ("A", "B", "C", "D"):
                    return ans
            # fallback: look for bare letter
            m2 = re.search(r'\b([ABCD])\b', raw)
            if m2:
                return m2.group(1)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None


def evaluate_one(qa_file: Path, model: str) -> dict:
    qa = json.loads(qa_file.read_text(encoding="utf-8"))
    sample_id  = qa_file.stem
    category   = qa.get("category", "unknown")
    task_type  = qa.get("task_type", "unknown")
    question   = qa["question"]
    options    = qa["options"]
    answer_gt  = qa["answer"]
    image_path = qa["input_image"]

    if not Path(image_path).exists():
        return {"sample_id": sample_id, "category": category, "task_type": task_type,
                "gt": answer_gt, "pred": None, "correct": False, "error": "image not found"}

    pred = call_model(model, question, options, image_path)
    correct = (pred == answer_gt) if pred is not None else False

    return {
        "sample_id":  sample_id,
        "category":   category,
        "task_type":  task_type,
        "gt":         answer_gt,
        "pred":       pred,
        "correct":    correct,
        "error":      None if pred is not None else "parse_failed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True, help="Model name to evaluate")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace(".", "_")
    out_file   = EVAL_DIR / f"{model_slug}.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-done results for resume
    done_results: dict = {}
    if out_file.exists():
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        done_results = {r["sample_id"]: r for r in existing.get("results", [])}
        print(f"Resuming: {len(done_results)} already done")

    qa_files = sorted(REVIEWED_DIR.glob("VRB_*.json"))
    if args.max_samples > 0:
        qa_files = qa_files[:args.max_samples]

    pending = [f for f in qa_files if f.stem not in done_results]
    print(f"Model: {args.model}")
    print(f"Total QAs: {len(qa_files)}  |  Pending: {len(pending)}  |  Workers: {args.num_workers}")

    results = list(done_results.values())
    lock = threading.Lock()
    done_count = len(results)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(evaluate_one, f, args.model): f for f in pending}
        for future in as_completed(futures):
            r = future.result()
            with lock:
                results.append(r)
                done_count += 1
                status = "✓" if r["correct"] else ("?" if r["error"] else "✗")
                if done_count % 50 == 0 or done_count == len(qa_files):
                    correct_so_far = sum(1 for x in results if x["correct"])
                    print(f"  [{done_count}/{len(qa_files)}] acc={correct_so_far/len(results)*100:.1f}%")
            # Save incrementally every 20
            if done_count % 20 == 0:
                _save(out_file, args.model, results, qa_files)

    _save(out_file, args.model, results, qa_files)
    _print_summary(args.model, results)


def _save(out_file: Path, model: str, results: list, qa_files):
    data = {"model": model, "total": len(qa_files), "results": results}
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_file)


def _print_summary(model: str, results: list):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors  = sum(1 for r in results if r["error"])

    print(f"\n{'='*55}")
    print(f"Model: {model}")
    print(f"{'='*55}")
    print(f"Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%  (errors: {errors})")

    # By category
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat_stats[r["category"]]["total"] += 1
        if r["correct"]:
            cat_stats[r["category"]]["correct"] += 1

    print(f"\nBy category:")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        print(f"  {cat:<28} {s['correct']:>3}/{s['total']:<3} = {s['correct']/s['total']*100:5.1f}%")

    # By task type
    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        task_stats[r["task_type"]]["total"] += 1
        if r["correct"]:
            task_stats[r["task_type"]]["correct"] += 1

    print(f"\nBy task type:")
    for t in sorted(task_stats.keys()):
        s = task_stats[t]
        print(f"  {t:<20} {s['correct']:>3}/{s['total']:<3} = {s['correct']/s['total']*100:5.1f}%")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()

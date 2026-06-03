#!/usr/bin/env python3
"""
OpenWorldQA Pipeline v2 — 5-Agent Architecture

Flow per video:
  SceneAnalyst → QuestionDesigner → DistractorForge → SmallModelProbe (×2) → Reviewer

Usage:
    python pipeline.py --frames_dir output/frames/ --output_dir output/reviewed/
    python pipeline.py --frames_dir output/frames/ --output_dir output/reviewed/ --num_workers 3
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import shutil
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# =============================================================================
# API Configuration — set these via environment variables or edit here.
# =============================================================================
API_BASE_URL  = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY       = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Export it before running: export OPENAI_API_KEY=your_key"
    )
MODEL_LARGE   = "gpt-5.4"     # SceneAnalyst, QuestionDesigner, DistractorForge, Reviewer
MODEL_SMALL   = "gpt-5-nano"  # SmallModelProbe
# =============================================================================

PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Category definitions (updated) ───────────────────────────────────────────

CATEGORIES = [
    "C1_fit_clearance",
    "C2_spatial",
    "C3_containment",
    "C4_support",
    "C5_friction",
    "C6_inertia",
    "C7_fluidity",
    "C8_deformability",
    "C9_tool_use",
    "C10_chain_reaction",
    "C11_process_race",
    "C12_multi_body",
]

CATEGORY_DESCS = {
    "C1_fit_clearance":   "Fit/gap prediction — will the object pass through or fit into a space?",
    "C2_spatial":         "Final spatial position or region after a physical event",
    "C3_containment":     "Overflow, underfill, fit vs. no-fit, liquid distribution",
    "C4_support":         "Stability, tipping direction, collapse, balance",
    "C5_friction":        "Sliding distance, slip vs. grip, stopping point on a surface",
    "C6_inertia":         "Trajectory, direction, or distance after force applied or removed",
    "C7_fluidity":        "Flow path, overflow threshold, splash pattern, absorption",
    "C8_deformability":   "Stretch, tear, crumple, snap, permanent vs. elastic deformation",
    "C9_tool_use":        "Physical result of tool acting on target object",
    "C10_chain_reaction": "Indirect / downstream effect beyond the direct contact point",
    "C11_process_race":   "Which of two concurrent physical processes completes first",
    "C12_multi_body":     "Momentum transfer, force contest, or spatial conflict between bodies",
}


# =============================================================================
# Core API helpers
# =============================================================================

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(prompt_text: str, image_paths: list[str],
             model: str = MODEL_LARGE) -> str:
    """Call the VLM API with a text prompt and zero or more images."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for img_path in image_paths:
        b64 = _encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object from model output (handles markdown fences)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = match.group(1) if match else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# =============================================================================
# Frame helpers
# =============================================================================

def load_frames(frames_dir: Path) -> tuple[list[str], dict]:
    """Return sorted list of frame paths and the metadata dict."""
    frames = sorted(frames_dir.glob("frame_*.jpg"), key=lambda p: p.name)
    frame_paths = [str(p) for p in frames]
    meta_path = frames_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return frame_paths, metadata


def resolve_anchor(frames_dir: Path, anchor_frame_name: str) -> str:
    """Return the absolute path for the anchor frame chosen by SceneAnalyst.
    Handles both 'frame_008' and 'frame_008.jpg' forms."""
    # Ensure .jpg extension
    name = anchor_frame_name if anchor_frame_name.endswith(".jpg") else anchor_frame_name + ".jpg"
    path = frames_dir / name
    if not path.exists():
        # Fallback: first frame
        fallback = sorted(frames_dir.glob("frame_*.jpg"))[0]
        return str(fallback)
    return str(path)


# =============================================================================
# Agent 1 — SceneAnalyst
# =============================================================================

def run_scene_analyst(task: dict) -> dict | None:
    """
    Receives ALL frames. Outputs structured scene report + selects anchor frame.
    Returns parsed JSON dict or None on failure.
    """
    frames_dir = Path(task["frames_dir"])
    frame_paths, _ = load_frames(frames_dir)

    prompt = (PROMPTS_DIR / "scene_analyst.txt").read_text(encoding="utf-8")
    prompt += f"\n\nFRAMES PROVIDED (in temporal order): {', '.join(Path(p).name for p in frame_paths)}"

    try:
        raw = call_vlm(prompt, frame_paths, model=MODEL_LARGE)
        result = extract_json(raw)
        if result is None:
            print(f"  [SceneAnalyst] JSON parse failed for {task['video_id']}")
            return None
        if "anchor_frame" not in result or "suitable_categories" not in result:
            print(f"  [SceneAnalyst] Missing required fields for {task['video_id']}")
            return None
        return result
    except Exception as e:
        print(f"  [SceneAnalyst] API error for {task['video_id']}: {e}")
        return None


# =============================================================================
# Agent 2 — QuestionDesigner
# =============================================================================

def run_question_designer(task: dict, scene_report: dict) -> dict | None:
    """
    Receives scene report (text only, no images).
    Outputs 6 question skeletons.
    """
    prompt = (PROMPTS_DIR / "question_designer.txt").read_text(encoding="utf-8")
    prompt += f"\n\nSCENE REPORT:\n{json.dumps(scene_report, indent=2, ensure_ascii=False)}"
    prompt += f"\n\nCURRENT CATEGORY DISTRIBUTION (accepted samples so far):\n"
    for cat in CATEGORIES:
        prompt += f"  {cat}: {task['category_distribution'].get(cat, 0)}\n"
    prompt += "\nPrioritise categories with the lowest count to keep the dataset balanced."

    try:
        raw = call_vlm(prompt, [], model=MODEL_LARGE)  # text-only
        result = extract_json(raw)
        if result is None or "question_skeletons" not in result:
            print(f"  [QuestionDesigner] Parse failed for {task['video_id']}")
            return None
        return result
    except Exception as e:
        print(f"  [QuestionDesigner] API error for {task['video_id']}: {e}")
        return None


# =============================================================================
# Agent 3 — DistractorForge
# =============================================================================

def run_distractor_forge(task: dict, scene_report: dict,
                         skeletons: dict) -> list[dict] | None:
    """
    Receives scene report + question skeletons (text only).
    Outputs 6 complete QA drafts (3 marked selected=true).
    """
    prompt = (PROMPTS_DIR / "distractor_forge.txt").read_text(encoding="utf-8")
    prompt += f"\n\nSCENE REPORT:\n{json.dumps(scene_report, indent=2, ensure_ascii=False)}"
    prompt += f"\n\nQUESTION SKELETONS:\n{json.dumps(skeletons, indent=2, ensure_ascii=False)}"

    try:
        raw = call_vlm(prompt, [], model=MODEL_LARGE)  # text-only
        result = extract_json(raw)
        if result is None or "qa_drafts" not in result:
            print(f"  [DistractorForge] Parse failed for {task['video_id']}")
            return None
        # Return only the 3 selected items
        selected = [qa for qa in result["qa_drafts"] if qa.get("selected") is True]
        if not selected:
            # Fallback: take first 3
            selected = result["qa_drafts"][:3]
        return selected
    except Exception as e:
        print(f"  [DistractorForge] API error for {task['video_id']}: {e}")
        return None


# =============================================================================
# Agent 4 — SmallModelProbe  (runs twice per QA with shuffled options)
# =============================================================================

OPTION_KEYS = ["A", "B", "C", "D"]


def _shuffle_options(qa: dict) -> tuple[dict, dict]:
    """
    Return (shuffled_qa, key_mapping) where key_mapping maps new_key → original_key.
    The answer field in shuffled_qa is updated to the new key of the correct answer.
    """
    options = qa["options"]
    original_answer = qa["answer"]

    keys = list(options.keys())
    shuffled_keys = keys[:]
    random.shuffle(shuffled_keys)

    # new_key = shuffled_keys[i], old_key = keys[i]
    key_mapping = {shuffled_keys[i]: keys[i] for i in range(len(keys))}
    reverse_mapping = {v: k for k, v in key_mapping.items()}  # old→new

    new_options = {shuffled_keys[i]: options[keys[i]] for i in range(len(keys))}
    new_answer = reverse_mapping[original_answer]

    shuffled_qa = {**qa, "options": new_options, "answer": new_answer}
    return shuffled_qa, reverse_mapping


def _build_probe_prompt(qa: dict) -> str:
    """Build the full text prompt for the small model probe."""
    base = (PROMPTS_DIR / "probe_prompt.txt").read_text(encoding="utf-8")
    options_text = "\n".join(f"  {k}: {v}" for k, v in qa["options"].items())
    return f"{base}\n\nQUESTION: {qa['question']}\n\nOPTIONS:\n{options_text}"


def run_small_model_probe(task: dict, qa: dict, anchor_path: str) -> str:
    """
    Run the probe TWICE with shuffled option order.
    Each attempt retries up to MAX_PROBE_RETRIES times on API error.
    Returns "too_easy" if the small model gets the correct answer on BOTH attempts,
    otherwise "hard_enough".
    """
    MAX_PROBE_RETRIES = 3   # max retries per attempt on API error
    RETRY_SLEEP = 3         # seconds between retries

    video_id = task["video_id"]
    results = []

    for attempt in range(2):
        shuffled_qa, reverse_map = _shuffle_options(qa)
        prompt = _build_probe_prompt(shuffled_qa)
        got_answer = False

        for retry in range(MAX_PROBE_RETRIES):
            try:
                raw = call_vlm(prompt, [anchor_path], model=MODEL_SMALL)
                parsed = extract_json(raw)
                if parsed is None:
                    # Unparseable response — treat as wrong, no retry needed
                    results.append(False)
                    got_answer = True
                    break
                probe_answer_new_key = parsed.get("answer", "")
                probe_answer_orig_key = reverse_map.get(probe_answer_new_key, "")
                is_correct = (probe_answer_orig_key == qa["answer"])
                results.append(is_correct)
                got_answer = True
                break
            except Exception as e:
                err_str = str(e)
                print(f"  [Probe] attempt {attempt+1} retry {retry+1}/{MAX_PROBE_RETRIES} "
                      f"for {video_id}: {e}")
                if retry < MAX_PROBE_RETRIES - 1:
                    time.sleep(RETRY_SLEEP)

        if not got_answer:
            # All retries exhausted — treat as wrong (don't discard QA due to API flakiness)
            print(f"  [Probe] attempt {attempt+1} for {video_id}: "
                  f"all retries failed, treating as hard_enough")
            results.append(False)

    if all(results):
        return "too_easy"
    return "hard_enough"


# =============================================================================
# Agent 5 — Reviewer
# =============================================================================

def run_reviewer(task: dict, qa_list: list[dict],
                 anchor_path: str, context_paths: list[str],
                 max_retries: int = 1) -> list[dict] | None:
    """
    Receives anchor + all context frames + QA items that passed the probe.
    Returns review_list or None on failure.
    """
    prompt = (PROMPTS_DIR / "review_prompt.txt").read_text(encoding="utf-8")
    context_names = [Path(p).name for p in context_paths]
    prompt += (
        f"\n\nINPUT:"
        f"\n  [ANCHOR]  Image 1 = {Path(anchor_path).name}  ← anchor frame (test model sees ONLY this)"
        f"\n  [CONTEXT] Images 2-{1+len(context_paths)} = {', '.join(context_names)}  ← reference only"
        f"\n\nQA BATCH:\n{json.dumps({'qa_list': qa_list}, indent=2, ensure_ascii=False)}"
    )

    # Anchor first, then context — Image 1 is always the anchor
    all_images = [anchor_path] + context_paths

    for attempt in range(max_retries + 1):
        try:
            raw = call_vlm(prompt, all_images, model=MODEL_LARGE)
            result = extract_json(raw)
            if result is None or "review_list" not in result:
                print(f"  [Reviewer] Parse failed for {task['video_id']} (attempt {attempt+1})")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return None
            return result["review_list"]
        except Exception as e:
            print(f"  [Reviewer] API error for {task['video_id']} (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
    return None


# =============================================================================
# Full pipeline for one video
# =============================================================================

def run_single_video(task: dict, args) -> dict:
    video_id = task["video_id"]
    start = time.time()
    frames_dir = Path(task["frames_dir"])
    print(f"\n[Worker] ── {video_id} ──────────────────")

    frame_paths, _ = load_frames(frames_dir)
    if not frame_paths:
        return {"video_id": video_id, "status": "failed",
                "reason": "No frames found", "duration": 0}

    # ── Agent 1: SceneAnalyst ──────────────────────────────────────────────
    t0 = time.time()
    scene_report = run_scene_analyst(task)
    if scene_report is None:
        _move_done(task)
        return _fail(video_id, "SceneAnalyst failed", start)
    anchor_name = scene_report.get("anchor_frame", Path(frame_paths[0]).name)
    anchor_path = resolve_anchor(frames_dir, anchor_name)
    # Normalise anchor_name to include .jpg extension for index lookup
    anchor_basename = anchor_name if anchor_name.endswith(".jpg") else anchor_name + ".jpg"
    anchor_idx = next((i for i, p in enumerate(frame_paths)
                       if Path(p).name == anchor_basename), 0)
    # Context = only frames AFTER the anchor (temporally), sorted by name.
    # Frames before the anchor are excluded to avoid confusing the Reviewer
    # about which image is the "initial condition" frame.
    context_paths = frame_paths[anchor_idx + 1:]  # post-anchor frames only
    print(f"  [1/5] SceneAnalyst  {time.time()-t0:.1f}s  anchor={anchor_name}  "
          f"context_frames={len(context_paths)}  "
          f"categories={scene_report.get('suitable_categories', [])}")

    # ── Agent 2: QuestionDesigner ──────────────────────────────────────────
    t0 = time.time()
    skeletons = run_question_designer(task, scene_report)
    if skeletons is None:
        _move_done(task)
        return _fail(video_id, "QuestionDesigner failed", start)
    print(f"  [2/5] QuestionDesigner  {time.time()-t0:.1f}s  "
          f"{len(skeletons.get('question_skeletons', []))} skeletons")

    # ── Agent 3: DistractorForge ───────────────────────────────────────────
    t0 = time.time()
    qa_drafts = run_distractor_forge(task, scene_report, skeletons)
    if not qa_drafts:
        _move_done(task)
        return _fail(video_id, "DistractorForge failed", start)
    print(f"  [3/5] DistractorForge  {time.time()-t0:.1f}s  "
          f"{len(qa_drafts)} QA drafts selected")

    # Attach video metadata to each draft
    for qa in qa_drafts:
        qa["video_path"] = str(frames_dir)
        qa["anchor_frame"] = anchor_name
        if qa.get("category") not in CATEGORIES:
            qa["category"] = "unknown"

    # Optionally save intermediate drafts
    if args.save_generated:
        gen_dir = Path(args.output_dir).parent / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (gen_dir / f"{video_id}_{ts}.json").write_text(
            json.dumps({"scene_report": scene_report,
                        "skeletons": skeletons,
                        "qa_drafts": qa_drafts}, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ── Agent 4: SmallModelProbe (×2 per QA) ──────────────────────────────
    t0 = time.time()
    surviving_qas = []
    probe_results = []
    for qa in qa_drafts:
        verdict = run_small_model_probe(task, qa, anchor_path)
        probe_results.append(verdict)
        if verdict == "hard_enough":
            surviving_qas.append(qa)

    print(f"  [4/5] SmallModelProbe  {time.time()-t0:.1f}s  "
          f"results={probe_results}  surviving={len(surviving_qas)}/{len(qa_drafts)}")

    if not surviving_qas:
        print(f"  [4/5] All QAs too easy — discarding {video_id}")
        if args.save_rejections:
            _save_rejection(task, args, qa_drafts,
                            [{"decision": "reject", "score": 0,
                              "suggestions": "Probe: answered correctly on both attempts"}
                             for _ in qa_drafts])
        _move_done(task)
        return {
            "video_id": video_id, "status": "rejected",
            "accepted_count": 0,
            "reason": "All QAs passed probe (too easy)",
            "duration": time.time() - start,
        }

    # ── Agent 5: Reviewer ──────────────────────────────────────────────────
    t0 = time.time()
    review_list = run_reviewer(task, surviving_qas, anchor_path, context_paths)
    if review_list is None:
        _move_done(task)
        return _fail(video_id, "Reviewer failed", start)

    # Pad if Reviewer returned fewer entries than expected
    while len(review_list) < len(surviving_qas):
        review_list.append({"decision": "reject", "score": 0,
                             "suggestions": "Review response truncated"})

    decisions = [r.get("decision") for r in review_list[:len(surviving_qas)]]
    print(f"  [5/5] Reviewer  {time.time()-t0:.1f}s  decisions={decisions}")

    # ── Save accepted / rejected ───────────────────────────────────────────
    total = time.time() - start
    out_dir = Path(args.output_dir)
    rej_dir = Path(args.output_dir).parent / "rejections"
    accepted_ids: list[str] = []
    accepted_categories: list[str] = []

    for qa, review in zip(surviving_qas, review_list):
        category = qa.get("category", "unknown")
        # Skip QAs with invalid/unknown category to avoid KeyError in seq_counters
        if category not in CATEGORIES:
            print(f"  [Worker] {video_id}: skipping QA with invalid category '{category}'")
            continue
        if review.get("decision") == "accept" and review.get("score", 0) >= 7:
            sample_id = task["queue"].claim_sample_id(category)
            qa["sample_id"] = sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            final = {
                **qa,
                "_review": review,
                "_pipeline_metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "duration_seconds": round(total, 2),
                    "source_video_id": video_id,
                    "anchor_frame": anchor_name,
                    "pipeline_version": "v2",
                },
            }
            out_file = out_dir / f"{sample_id}.json"
            out_file.write_text(json.dumps(final, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            accepted_ids.append(sample_id)
            accepted_categories.append(category)
            print(f"  ✓  {sample_id}  ({category})  score={review.get('score')}")
        else:
            print(f"  ✗  {category}  REJECTED  score={review.get('score')}")
            if args.save_rejections:
                _save_rejection(task, args, [qa], [review])

    _move_done(task)

    if accepted_ids:
        return {
            "video_id": video_id,
            "status": "accepted",
            "accepted_count": len(accepted_ids),
            "sample_ids": accepted_ids,
            "categories": accepted_categories,
            "duration": total,
        }
    return {
        "video_id": video_id,
        "status": "rejected",
        "accepted_count": 0,
        "reason": "All surviving QAs rejected by Reviewer",
        "duration": total,
    }


# =============================================================================
# Helpers
# =============================================================================

def _fail(video_id: str, reason: str, start: float) -> dict:
    return {"video_id": video_id, "status": "failed",
            "reason": reason, "duration": time.time() - start}


def _move_done(task: dict):
    src = task["frames_dir"]
    dst = str(Path(task["done_dir"]) / task["video_id"])
    try:
        shutil.move(src, dst)
    except Exception:
        pass


def _save_rejection(task: dict, args, qa_list: list[dict],
                    review_list: list[dict]):
    rej_dir = Path(args.output_dir).parent / "rejections"
    rej_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_serializable = {k: v for k, v in task.items()
                         if k not in ("queue", "done_dir")}
    (rej_dir / f"{task['video_id']}_{ts}_rejected.json").write_text(
        json.dumps({"task": task_serializable,
                    "qa_list": qa_list,
                    "review_list": review_list,
                    "timestamp": datetime.now().isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")


# =============================================================================
# Task queue
# =============================================================================

class VideoQueue:
    def __init__(self, dist_path: str | None = None):
        self._lock = threading.Lock()
        self._pending: list[dict] = []
        self._processing: set[str] = set()
        self._completed: set[str] = set()
        self._rejected: set[str] = set()
        self._failed: set[str] = set()
        self._category_counter = {c: 0 for c in CATEGORIES}
        self._seq_counters     = {c: 0 for c in CATEGORIES}
        self._dist_path = Path(dist_path) if dist_path else None

    def load_videos(self, frames_dir: str, output_dir: str):
        frames_path = Path(frames_dir)
        self._done_dir = Path(output_dir).parent / "done"
        self._done_dir.mkdir(parents=True, exist_ok=True)

        # Resume seq counters from existing accepted files
        out_path = Path(output_dir)
        if out_path.exists():
            for f in out_path.glob("OpenWorldQA_*.json"):
                m = re.match(r"OpenWorldQA_([A-Za-z0-9_]+)_(\d+)\.json", f.name)
                if m:
                    cat, seq = m.group(1), int(m.group(2))
                    if cat in self._seq_counters and seq > self._seq_counters[cat]:
                        self._seq_counters[cat] = seq
                        self._category_counter[cat] = seq
            existing = {c: v for c, v in self._seq_counters.items() if v > 0}
            if existing:
                print(f"Resuming seq counters: {existing}")

        video_dirs = [d for d in frames_path.iterdir()
                      if d.is_dir() and any(d.glob("frame_*.jpg"))]
        print(f"Found {len(video_dirs)} frame directories in {frames_dir}")

        ready, skipped = [], 0
        for vd in video_dirs:
            if (self._done_dir / vd.name).exists():
                skipped += 1
                continue
            ready.append({"video_id": vd.name, "frames_dir": str(vd)})

        with self._lock:
            self._pending = ready
        print(f"Ready: {len(ready)} videos  (skipped {skipped} already done)")

    def get_next_task(self) -> dict | None:
        with self._lock:
            if not self._pending:
                return None
            task = self._pending.pop(0)
            task["category_distribution"] = dict(self._category_counter)
            task["queue"]    = self
            task["done_dir"] = str(self._done_dir)
            self._processing.add(task["video_id"])
            return task

    def claim_sample_id(self, category: str) -> str:
        with self._lock:
            self._seq_counters[category] += 1
            self._category_counter[category] += 1
            self._write_distribution()
            return f"OpenWorldQA_{category}_{self._seq_counters[category]:04d}"

    def mark_done(self, video_id: str, status: str = "completed"):
        with self._lock:
            self._processing.discard(video_id)
            if status == "completed":
                self._completed.add(video_id)
            elif status == "rejected":
                self._rejected.add(video_id)
            else:
                self._failed.add(video_id)

    def _write_distribution(self):
        if self._dist_path is None:
            return
        try:
            self._dist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._dist_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._category_counter, indent=2, ensure_ascii=False),
                encoding="utf-8")
            tmp.replace(self._dist_path)
        except Exception as e:
            print(f"[Warning] Failed to write distribution: {e}")

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending":    len(self._pending),
                "processing": len(self._processing),
                "completed":  len(self._completed),
                "rejected":   len(self._rejected),
                "failed":     len(self._failed),
                "categories": dict(self._category_counter),
            }


# =============================================================================
# Main scheduler
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OpenWorldQA Pipeline v2 — 5-Agent Architecture")
    parser.add_argument("--frames_dir",      default="output/frames/")
    parser.add_argument("--output_dir",      default="output/reviewed/")
    parser.add_argument("--num_workers",     type=int, default=3)
    parser.add_argument("--max_videos",      type=int, default=0)
    parser.add_argument("--save_rejections", action="store_true")
    parser.add_argument("--save_generated",  action="store_true")
    parser.add_argument("--dry_run",         action="store_true")
    parser.add_argument("--dist_file",
                        default="output/category_distribution.json")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenWorldQA Pipeline v2")
    print("=" * 60)
    print(f"Large model : {MODEL_LARGE}")
    print(f"Small model : {MODEL_SMALL}")
    print(f"Workers     : {args.num_workers}")
    print(f"Frames dir  : {args.frames_dir}")
    print(f"Output dir  : {args.output_dir}")
    print("=" * 60)

    queue = VideoQueue(dist_path=args.dist_file)
    queue.load_videos(args.frames_dir, args.output_dir)

    s = queue.stats()
    if s["pending"] == 0:
        print("\nNo videos ready. Run extract_frames.py first.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {s['pending']} videos.")
        return

    if args.max_videos > 0:
        with queue._lock:
            queue._pending = queue._pending[:args.max_videos]

    results = []
    lock = threading.Lock()
    start_time = time.time()
    total_qa_accepted = 0

    print(f"\nStarting {args.num_workers} parallel workers...\n{'='*60}\n")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures: dict = {}

        def _fill():
            while len(futures) < args.num_workers:
                task = queue.get_next_task()
                if task is None:
                    break
                f = executor.submit(run_single_video, task, args)
                futures[f] = task["video_id"]

        _fill()

        while futures:
            done_iter = as_completed(list(futures))
            f = next(done_iter)
            vid = futures.pop(f)
            result = f.result()

            with lock:
                results.append(result)
                total_qa_accepted += result.get("accepted_count", 0)

            queue.mark_done(vid, status="completed" if result["status"] == "accepted"
                            else result["status"]
                            if result["status"] in ("rejected", "failed")
                            else "failed")

            st = queue.stats()
            done_total = st["completed"] + st["rejected"] + st["failed"]
            elapsed = time.time() - start_time
            rate = done_total / elapsed * 60 if elapsed > 0 else 0
            print(f"\n{'─'*60}")
            print(f"[Progress] {done_total} done  "
                  f"({st['completed']}✓ {st['rejected']}✗ {st['failed']}⚠)  "
                  f"| {total_qa_accepted} QAs saved | {st['pending']} remaining")
            print(f"[Rate] {rate:.1f} videos/min  |  Active: {len(futures)}")
            print(f"{'─'*60}\n")
            sys.stdout.flush()
            _fill()

    # Summary
    total_time = time.time() - start_time
    st = queue.stats()
    done_total = st["completed"] + st["rejected"] + st["failed"]
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time       : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Videos processed : {done_total}")
    print(f"  Accepted (≥1 QA) : {st['completed']}")
    print(f"  All rejected     : {st['rejected']}")
    print(f"  Failed           : {st['failed']}")
    print(f"Total QAs saved  : {total_qa_accepted}")
    if done_total > 0 and total_time > 0:
        print(f"Avg rate         : {done_total/total_time*60:.1f} videos/min")
    print("\nCategory distribution (accepted QA items):")
    for cat, count in sorted(st["categories"].items()):
        if count > 0:
            bar = "█" * min(count, 40)
            print(f"  {cat:<22} {bar} {count}")
    print(f"\nResults  : {args.output_dir}")
    print(f"Dist file: {args.dist_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

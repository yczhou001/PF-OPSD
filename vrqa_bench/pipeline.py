#!/usr/bin/env python3
"""
VRBench VQA Pipeline v2 — Programmatic Ground Truth

Changelog (v1 -> v2):
  - Removed maze3d (isometric view direction recognition is unreliable)
  - Removed video frame extraction / SolutionAnalyst (VLM hallucinates paths from frames)
  - Switched to programmatic solving via state.json: maze=BFS, sokoban=push BFS, irregular_maze=geometry angles
  - VLM only writes the question description text (no longer generates answers)
  - Reviewer only checks question text quality (answers are guaranteed by the program)

Usage:
    python pipeline.py --num_workers 3
    python pipeline.py --task maze --num_workers 3
    python pipeline.py --max_samples 5 --save_rejections
"""

import csv
import json
import math
import os
import random
import re
import sys
import time
import argparse
import threading
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# =============================================================================
# API Configuration
# =============================================================================
API_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_KEY      = os.environ.get("OPENAI_API_KEY", "")
MODEL_LARGE  = "gpt-5.5"
MODEL_SMALL  = "gpt-5.4-nano"
# =============================================================================
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

PROJECT_ROOT = Path(__file__).parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROMPTS_DIR  = PROJECT_ROOT / "prompts"
OUTPUT_DIR   = PROJECT_ROOT / "output"

TASKS = ["maze", "irregular_maze", "sokoban"]

CATEGORIES = [
    "C1_turn_count",
    "C2_turn_direction",
    "C4_sokoban_push",
    "C5_direction_count",
    "C6_push_dir_count",
]

TASK_CATEGORIES = {
    "maze":           ["C1_turn_count", "C2_turn_direction", "C5_direction_count"],
    "irregular_maze": ["C1_turn_count", "C2_turn_direction"],
    "sokoban":        ["C4_sokoban_push", "C6_push_dir_count"],
}

DIRECTIONS_2D = ["up", "down", "left", "right"]


# =============================================================================
# API helpers
# =============================================================================

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(prompt_text: str, image_paths: list[str],
             model: str = MODEL_LARGE) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    content: list[dict] = [{"type": "text", "text": prompt_text}]
    for img_path in image_paths:
        b64  = encode_image(img_path)
        ext  = Path(img_path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def extract_json(text: str) -> dict | None:
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
# Path helpers
# =============================================================================

def resolve_path(csv_path: str) -> str:
    parts = csv_path.split("/")
    if parts[0] in ("eval", "train"):
        parts = parts[1:]
    return str(RAW_DATA_DIR / Path(*parts))


def load_state(video_path: str) -> dict | None:
    """
    state.json and video share the same parent directory:
      videos/hard_0097_0.mp4  ->  states/hard_0097.json
    """
    p = Path(video_path)
    base_stem  = re.sub(r"_\d+$", "", p.stem)   # hard_0097_0 -> hard_0097
    state_path = p.parent.parent / "states" / f"{base_stem}.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# =============================================================================
# Programmatic solver — maze (BFS)
# =============================================================================

def _count_lr_turns(steps: list[str]) -> tuple[int, int]:
    """Count left and right turns in a step sequence (2D cardinal directions)."""
    dvec = {
        "up":    (-1,  0),
        "down":  ( 1,  0),
        "left":  ( 0, -1),
        "right": ( 0,  1),
    }
    left = right = 0
    for i in range(1, len(steps)):
        if steps[i] == steps[i - 1]:
            continue
        a = dvec.get(steps[i - 1])
        b = dvec.get(steps[i])
        if not a or not b:
            continue
        cross = a[0] * b[1] - a[1] * b[0]
        if cross > 0:
            right += 1
        elif cross < 0:
            left += 1
        else:
            # 180 degree U-turn -> counts as 2
            left  += 1
            right += 1
    return left, right


def solve_maze(state: dict) -> dict | None:
    """BFS on the state grid. Returns exact solution stats."""
    grid = state["grid"]["data"]
    H, W = len(grid), len(grid[0])
    sr = state["entities"]["player"]["grid_pos"]["row"]
    sc = state["entities"]["player"]["grid_pos"]["col"]
    gr = state["entities"]["goal"]["grid_pos"]["row"]
    gc = state["entities"]["goal"]["grid_pos"]["col"]

    DIRS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]
    queue   = deque([((sr, sc), [])])
    visited = {(sr, sc)}

    while queue:
        (r, c), path = queue.popleft()
        if r == gr and c == gc:
            turns       = sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])
            left, right = _count_lr_turns(path)
            return {
                "steps":        path,
                "total_turns":  turns,
                "left_turns":   left,
                "right_turns":  right,
                "start_grid":   (sr, sc),
                "goal_grid":    (gr, gc),
                "total_steps":  len(path),
            }
        for dr, dc, name in DIRS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and grid[nr][nc] != 1
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [name]))
    return None


# =============================================================================
# Programmatic solver — sokoban (minimum-push BFS)
# =============================================================================

def solve_sokoban(state: dict) -> dict | None:
    """
    BFS over (player_pos, box_positions) states, minimising push count.
    Uses inner reachability check so the outer BFS expands by push, not by move.
    """
    grid = state["grid"]["data"]
    H, W = len(grid), len(grid[0])

    player = (state["entities"]["player"]["grid_pos"]["row"],
              state["entities"]["player"]["grid_pos"]["col"])
    goal   = (state["entities"]["goal"]["grid_pos"]["row"],
              state["entities"]["goal"]["grid_pos"]["col"])
    boxes  = tuple(sorted(
        (b["grid_pos"]["row"], b["grid_pos"]["col"])
        for b in state["entities"]["boxes"]
    ))
    goals  = tuple(sorted([goal] * len(boxes)))

    def is_wall(r: int, c: int) -> bool:
        return not (0 <= r < H and 0 <= c < W) or grid[r][c] == 1

    def player_can_reach(src: tuple, dst: tuple, box_set: frozenset) -> bool:
        if src == dst:
            return True
        visited_r: set = {src}
        q: deque = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb not in visited_r and not is_wall(*nb) and nb not in box_set:
                    if nb == dst:
                        return True
                    visited_r.add(nb)
                    q.append(nb)
        return False

    DIRS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]

    init_state = (player, boxes)
    queue   = deque([(init_state, [])])
    visited = {init_state}

    while queue:
        (pl, bs), pushes = queue.popleft()

        if bs == goals:
            push_dirs = [d for _, _, d in pushes]
            return {
                "push_sequence":        push_dirs,
                "total_pushes":         len(push_dirs),
                "first_push_direction": push_dirs[0] if push_dirs else None,
            }

        bs_set = frozenset(bs)
        for dr, dc, dname in DIRS:
            for box in bs:
                req_player = (box[0] - dr, box[1] - dc)
                new_box    = (box[0] + dr, box[1] + dc)
                if is_wall(*req_player) or is_wall(*new_box):
                    continue
                if new_box in bs_set:
                    continue
                if not player_can_reach(pl, req_player, bs_set):
                    continue
                new_pl = box
                new_bs = tuple(sorted(b if b != box else new_box for b in bs))
                new_state = (new_pl, new_bs)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, pushes + [(dr, dc, dname)]))

    return None


# =============================================================================
# Programmatic solver — irregular_maze (geometry)
# =============================================================================

def solve_irregular_maze(state: dict) -> dict | None:
    """
    Count direction changes and left/right turns at interior junction points.
    Uses vector angles; a junction is a turn when angle > 30°.
    In image coordinates (y-axis down): cross > 0 → right, cross < 0 → left.
    """
    meta          = state.get("metadata", {})
    solution_path = meta.get("solution_path")
    if not solution_path or len(solution_path) < 2:
        return None

    STRAIGHT_THRESHOLD = 30.0
    direction_changes = left_turns = right_turns = 0

    for i in range(1, len(solution_path) - 1):
        p0, p1, p2 = solution_path[i - 1], solution_path[i], solution_path[i + 1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            continue
        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.degrees(math.acos(cos_a))
        if angle <= STRAIGHT_THRESHOLD:
            continue
        direction_changes += 1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross > 0:
            right_turns += 1
        else:
            left_turns += 1

    return {
        "direction_changes": direction_changes,
        "left_turns":        left_turns,
        "right_turns":       right_turns,
        "solution_path":     solution_path,
    }


def solve_puzzle(task_type: str, state: dict) -> dict | None:
    if task_type == "maze":
        return solve_maze(state)
    if task_type == "sokoban":
        return solve_sokoban(state)
    if task_type == "irregular_maze":
        return solve_irregular_maze(state)
    return None


# =============================================================================
# Distractor + option generation (programmatic, no VLM)
# =============================================================================

def _make_count_options(correct: int) -> tuple[dict, str]:
    """
    Generate 4 options for a count question.
    Distractors are numerically close (within ±4) and all ≥ 0.
    Correct answer is placed at a random letter (A/B/C/D).
    """
    pool: list[int] = []
    for delta in range(1, 10):
        if correct - delta >= 0:
            pool.append(correct - delta)
        pool.append(correct + delta)
        if len(pool) >= 9:
            break
    preferred = [c for c in pool if abs(c - correct) <= 4]
    use_pool  = preferred if len(preferred) >= 3 else pool
    distractors = random.sample(use_pool[:min(8, len(use_pool))], 3)

    all_opts = distractors + [correct]
    random.shuffle(all_opts)
    keys    = ["A", "B", "C", "D"]
    options = {keys[i]: str(all_opts[i]) for i in range(4)}
    answer  = keys[all_opts.index(correct)]
    return options, answer


def _make_direction_options(correct: str) -> tuple[dict, str]:
    """All 4 cardinal directions as options; correct placed at random letter."""
    dirs = list(DIRECTIONS_2D)
    random.shuffle(dirs)
    keys    = ["A", "B", "C", "D"]
    options = {keys[i]: dirs[i] for i in range(4)}
    answer  = keys[dirs.index(correct)]
    return options, answer


def build_qa_items(task_type: str, solution: dict) -> list[dict]:
    """
    Build QA items with programmatically verified answers.
    question field is left empty — QuestionWriter fills it in.
    """
    items: list[dict] = []

    if task_type == "maze":
        # C1: total direction changes
        opts, ans = _make_count_options(solution["total_turns"])
        items.append({
            "category":      "C1_turn_count",
            "options":       opts,
            "answer":        ans,
            "correct_value": str(solution["total_turns"]),
            "question":      "",
        })
        # C2: left OR right turns (random choice)
        ask_left = random.choice([True, False])
        val      = solution["left_turns"] if ask_left else solution["right_turns"]
        opts, ans = _make_count_options(val)
        items.append({
            "category":      "C2_turn_direction",
            "ask_left":      ask_left,
            "options":       opts,
            "answer":        ans,
            "correct_value": str(val),
            "question":      "",
        })
        # C5: steps in most-frequent direction (avoids trivially-zero answers)
        steps = solution["steps"]
        from collections import Counter
        counts = Counter(steps)
        ask_dir = counts.most_common(1)[0][0]  # direction with most steps
        dir_count = counts[ask_dir]
        opts, ans = _make_count_options(dir_count)
        items.append({
            "category":      "C5_direction_count",
            "ask_dir":       ask_dir,
            "options":       opts,
            "answer":        ans,
            "correct_value": str(dir_count),
            "question":      "",
        })

    elif task_type == "irregular_maze":
        # C1: direction changes at junctions
        opts, ans = _make_count_options(solution["direction_changes"])
        items.append({
            "category":      "C1_turn_count",
            "options":       opts,
            "answer":        ans,
            "correct_value": str(solution["direction_changes"]),
            "question":      "",
        })
        # C2: left OR right junction turns (random choice)
        ask_left = random.choice([True, False])
        val      = solution["left_turns"] if ask_left else solution["right_turns"]
        opts, ans = _make_count_options(val)
        items.append({
            "category":      "C2_turn_direction",
            "ask_left":      ask_left,
            "options":       opts,
            "answer":        ans,
            "correct_value": str(val),
            "question":      "",
        })

    elif task_type == "sokoban":
        # C4a: minimum push count
        opts, ans = _make_count_options(solution["total_pushes"])
        items.append({
            "category":      "C4_sokoban_push",
            "sub_type":      "count",
            "options":       opts,
            "answer":        ans,
            "correct_value": str(solution["total_pushes"]),
            "question":      "",
        })
        # C4b: first push direction
        if solution.get("first_push_direction"):
            opts, ans = _make_direction_options(solution["first_push_direction"])
            items.append({
                "category":      "C4_sokoban_push",
                "sub_type":      "direction",
                "options":       opts,
                "answer":        ans,
                "correct_value": solution["first_push_direction"],
                "question":      "",
            })
        # C6: pushes in most-frequent direction (avoids zero-count answers)
        push_seq = solution["push_sequence"]
        if push_seq:
            from collections import Counter
            push_counts = Counter(push_seq)
            ask_dir     = push_counts.most_common(1)[0][0]
            dir_count   = push_counts[ask_dir]
            opts, ans   = _make_count_options(dir_count)
            items.append({
                "category":      "C6_push_dir_count",
                "ask_dir":       ask_dir,
                "options":       opts,
                "answer":        ans,
                "correct_value": str(dir_count),
                "question":      "",
            })

    return items


# =============================================================================
# Agent — QuestionWriter  (VLM: writes question text only)
# =============================================================================

def run_question_writer(task: dict, qa_item: dict) -> str | None:
    prompt = (PROMPTS_DIR / "qa_forge.txt").read_text(encoding="utf-8")
    prompt += f"\n\nTASK TYPE: {task['task_type']}"
    prompt += f"\nCATEGORY: {qa_item['category']}"
    if "ask_left" in qa_item:
        prompt += f"\nDIRECTION: {'left' if qa_item['ask_left'] else 'right'}"
    if "sub_type" in qa_item:
        prompt += f"\nSUB_TYPE: {qa_item['sub_type']}"
    if "ask_dir" in qa_item:
        prompt += f"\nASK_DIRECTION: {qa_item['ask_dir']}"

    try:
        raw    = call_vlm(prompt, [task["image_path"]], model=MODEL_LARGE)
        result = extract_json(raw)
        if result and "question" in result:
            return result["question"]
        print(f"  [QuestionWriter] parse failed for {task['sample_id']}")
    except Exception as e:
        print(f"  [QuestionWriter] API error for {task['sample_id']}: {e}")
    return None


# =============================================================================
# Agent — SmallModelProbe  (unchanged logic, no video frames)
# =============================================================================

def _shuffle_options(qa: dict) -> tuple[dict, dict]:
    options          = qa["options"]
    original_answer  = qa["answer"]
    keys             = list(options.keys())
    shuffled_keys    = keys[:]
    random.shuffle(shuffled_keys)
    key_mapping      = {shuffled_keys[i]: keys[i] for i in range(len(keys))}
    reverse_mapping  = {v: k for k, v in key_mapping.items()}
    new_options      = {shuffled_keys[i]: options[keys[i]] for i in range(len(keys))}
    new_answer       = reverse_mapping[original_answer]
    return {**qa, "options": new_options, "answer": new_answer}, reverse_mapping


def run_small_model_probe(task: dict, qa: dict) -> str:
    """
    Present the question TWICE (with shuffled options) to the small model.
    Returns 'too_easy' only if it answers correctly on BOTH attempts.
    """
    MAX_RETRIES  = 3
    RETRY_SLEEP  = 3
    sample_id    = task["sample_id"]

    probe_base = (
        "You are given a spatial puzzle image. "
        "Answer the following multiple-choice question by choosing the single best option. "
        "Reply with JSON only: {\"answer\": \"A\"}\n"
        "The answer must be one of: A, B, C, or D."
    )

    results = []
    for attempt in range(2):
        shuffled_qa, reverse_map = _shuffle_options(qa)
        options_text = "\n".join(
            f"  {k}: {v}" for k, v in shuffled_qa["options"].items()
        )
        prompt = (
            f"{probe_base}\n\n"
            f"QUESTION: {shuffled_qa['question']}\n\n"
            f"OPTIONS:\n{options_text}"
        )
        got_answer = False
        for retry in range(MAX_RETRIES):
            try:
                raw    = call_vlm(prompt, [task["image_path"]], model=MODEL_SMALL)
                parsed = extract_json(raw)
                if parsed is None:
                    results.append(False)
                    got_answer = True
                    break
                probe_new  = parsed.get("answer", "")
                probe_orig = reverse_map.get(probe_new, "")
                results.append(probe_orig == qa["answer"])
                got_answer = True
                break
            except Exception as e:
                print(f"  [Probe] attempt {attempt+1} retry {retry+1}/{MAX_RETRIES} "
                      f"for {sample_id}: {e}")
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_SLEEP)
        if not got_answer:
            results.append(False)

    return "too_easy" if all(results) else "hard_enough"


# =============================================================================
# Agent — Reviewer  (simplified: only checks question text quality)
# =============================================================================

def run_reviewer(task: dict, qa_list: list[dict],
                 max_retries: int = 1) -> list[dict] | None:
    prompt  = (PROMPTS_DIR / "reviewer.txt").read_text(encoding="utf-8")
    prompt += f"\n\nTASK TYPE: {task['task_type']}"
    prompt += (
        f"\n\nQA LIST:\n"
        + json.dumps({"qa_list": qa_list}, indent=2, ensure_ascii=False)
    )

    for attempt in range(max_retries + 1):
        try:
            raw    = call_vlm(prompt, [task["image_path"]], model=MODEL_LARGE)
            result = extract_json(raw)
            if result and "review_list" in result:
                return result["review_list"]
            print(f"  [Reviewer] parse failed for {task['sample_id']} "
                  f"(attempt {attempt+1})")
        except Exception as e:
            print(f"  [Reviewer] API error for {task['sample_id']} "
                  f"(attempt {attempt+1}): {e}")
        if attempt < max_retries:
            time.sleep(2)
    return None


# =============================================================================
# Full pipeline for one sample
# =============================================================================

def run_single_sample(task: dict, args) -> dict:
    sample_id = task["sample_id"]
    start     = time.time()
    print(f"\n[Worker] ── {sample_id} ({task['task_type']}) ──────────────")

    # ── Step 1: load state.json ────────────────────────────────────────────
    state = load_state(task["video_path"])
    if state is None:
        return _fail(sample_id, "state.json not found", start)

    # ── Step 2: programmatic solve ─────────────────────────────────────────
    t0       = time.time()
    solution = solve_puzzle(task["task_type"], state)
    if solution is None:
        return _fail(sample_id, "Solver returned None (unsolvable?)", start)
    print(f"  [1/4] Solver  {time.time()-t0:.1f}s  "
          + (f"turns={solution.get('total_turns','?')}" if "total_turns" in solution
             else f"direction_changes={solution.get('direction_changes','?')}" if "direction_changes" in solution
             else f"pushes={solution.get('total_pushes','?')}"))

    # ── Step 3: build QA items (correct answers + options) ─────────────────
    qa_items = build_qa_items(task["task_type"], solution)
    if not qa_items:
        return _fail(sample_id, "build_qa_items returned empty", start)

    # ── Step 4: QuestionWriter — write question text ────────────────────────
    t0      = time.time()
    written = []
    for qa in qa_items:
        q = run_question_writer(task, qa)
        if q:
            qa["question"] = q
            written.append(qa)
    print(f"  [2/4] QuestionWriter  {time.time()-t0:.1f}s  "
          f"{len(written)}/{len(qa_items)} written")
    if not written:
        _move_done(task)
        return _fail(sample_id, "QuestionWriter failed for all items", start)

    # Save intermediate if requested
    if args.save_generated:
        gen_dir = OUTPUT_DIR / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (gen_dir / f"{sample_id}_{ts}.json").write_text(
            json.dumps({"solution": solution, "qa_list": written},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Step 5: SmallModelProbe ─────────────────────────────────────────────
    t0          = time.time()
    surviving   = []
    probe_results = []
    for qa in written:
        verdict = run_small_model_probe(task, qa)
        probe_results.append(verdict)
        if verdict == "hard_enough":
            surviving.append(qa)
    print(f"  [3/4] SmallModelProbe  {time.time()-t0:.1f}s  "
          f"results={probe_results}  surviving={len(surviving)}/{len(written)}")

    if not surviving:
        print(f"  [3/4] All QAs too easy — discarding {sample_id}")
        if args.save_rejections:
            for qa in written:
                _save_rejection(sample_id, qa,
                                {"decision": "reject", "score": 0,
                                 "suggestions": "Probe: answered correctly both attempts"})
        _move_done(task)
        return {"sample_id": sample_id, "status": "rejected",
                "accepted_count": 0, "reason": "All QAs too easy (probe)",
                "duration": time.time() - start}

    # ── Step 6: Reviewer ───────────────────────────────────────────────────
    t0          = time.time()
    review_list = run_reviewer(task, surviving)
    if review_list is None:
        _move_done(task)
        return _fail(sample_id, "Reviewer failed", start)

    while len(review_list) < len(surviving):
        review_list.append({"decision": "reject", "score": 0,
                             "suggestions": "Review response truncated"})

    decisions = [r.get("decision") for r in review_list[: len(surviving)]]
    print(f"  [4/4] Reviewer  {time.time()-t0:.1f}s  decisions={decisions}")

    # ── Step 7: save accepted ──────────────────────────────────────────────
    total        = time.time() - start
    accepted_ids: list[str] = []

    for qa, review in zip(surviving, review_list):
        category = qa.get("category", "unknown")
        if category not in CATEGORIES:
            continue

        if review.get("decision") == "accept" and review.get("score", 0) >= 7:
            out_id = task["queue"].claim_output_id(category)
            final  = {
                **{k: v for k, v in qa.items() if k not in ("ask_left",)},
                "source_sample_id": sample_id,
                "task_type":        task["task_type"],
                "input_image":      task["image_path"],
                "video_path":       task["video_path"],
                "_solution":        solution,
                "_review":          review,
                "_pipeline_metadata": {
                    "processed_at":    datetime.now().isoformat(),
                    "duration_seconds": round(total, 2),
                    "pipeline_version": "v2",
                },
            }
            out_file = OUTPUT_DIR / "reviewed" / f"{out_id}.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                json.dumps(final, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            accepted_ids.append(out_id)
            print(f"  ✓  {out_id}  score={review.get('score')}")
        else:
            print(f"  ✗  {category}  REJECTED  score={review.get('score')}")
            if args.save_rejections:
                _save_rejection(sample_id, qa, review)

    _move_done(task)

    if accepted_ids:
        return {"sample_id": sample_id, "status": "accepted",
                "accepted_count": len(accepted_ids), "output_ids": accepted_ids,
                "duration": total}
    return {"sample_id": sample_id, "status": "rejected",
            "accepted_count": 0, "reason": "All QAs rejected by Reviewer",
            "duration": total}


# =============================================================================
# Helpers
# =============================================================================

def _fail(sample_id: str, reason: str, start: float) -> dict:
    return {"sample_id": sample_id, "status": "failed",
            "reason": reason, "duration": time.time() - start}


def _move_done(task: dict):
    done_dir = OUTPUT_DIR / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / task["sample_id"]).touch()


def _save_rejection(sample_id: str, qa: dict, review: dict):
    rej_dir = OUTPUT_DIR / "rejections"
    rej_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (rej_dir / f"{sample_id}_{ts}_rejected.json").write_text(
        json.dumps({"sample_id": sample_id, "qa": qa, "review": review,
                    "timestamp": datetime.now().isoformat()},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# Sample Queue
# =============================================================================

class SampleQueue:
    def __init__(self):
        self._lock             = threading.Lock()
        self._pending:  list[dict] = []
        self._category_counter = {c: 0 for c in CATEGORIES}
        self._seq_counters     = {c: 0 for c in CATEGORIES}

    def load_samples(self, tasks_filter: list[str] | None = None,
                     split: str = "eval",
                     train_quota: dict | None = None):
        done_dir = OUTPUT_DIR / "done"
        done_ids = {p.name for p in done_dir.glob("*")} if done_dir.exists() else set()

        # Resume seq counters from existing output files
        reviewed_dir = OUTPUT_DIR / "reviewed"
        if reviewed_dir.exists():
            for f in reviewed_dir.glob("VRB_*.json"):
                m = re.match(r"VRB_([A-Za-z0-9_]+)_(\d+)\.json", f.name)
                if m:
                    cat, seq = m.group(1), int(m.group(2))
                    full_cat = next(
                        (c for c in CATEGORIES if c.endswith(cat) or c == cat), None
                    )
                    if full_cat and seq > self._seq_counters[full_cat]:
                        self._seq_counters[full_cat]     = seq
                        self._category_counter[full_cat] = seq

        meta_base      = RAW_DATA_DIR / "metadata" / split
        tasks_to_load  = tasks_filter or TASKS

        if split == "eval":
            difficulties_filter = ["hard"]
            quota_map: dict | None = None
        else:
            difficulties_filter = ["hard", "medium", "easy"]
            quota_map = train_quota or {"hard": 240, "medium": 100, "easy": 60}

        pool: dict[tuple, list] = {}
        for task_type in tasks_to_load:
            for diff in difficulties_filter:
                pool[(task_type, diff)] = []

        for folder in sorted(meta_base.iterdir()):
            parts = folder.name.rsplit("_", 2)
            if len(parts) != 3:
                continue
            t, variant, difficulty = parts
            if t not in tasks_to_load or difficulty not in difficulties_filter:
                continue

            csv_path = folder / "metadata.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            for row in rows:
                video_rel = row["video"]
                image_rel = row["input_image"]
                vid_stem  = Path(video_rel).stem
                sample_id = f"{t}_{variant}_{vid_stem}"

                if sample_id in done_ids:
                    continue

                pool[(t, difficulty)].append({
                    "sample_id":  sample_id,
                    "task_type":  t,
                    "variant":    variant,
                    "difficulty": difficulty,
                    "video_path": resolve_path(video_rel),
                    "image_path": resolve_path(image_rel),
                })

        samples = []
        rng = random.Random(42)
        for task_type in tasks_to_load:
            for diff in difficulties_filter:
                bucket = pool.get((task_type, diff), [])
                if quota_map:
                    quota  = quota_map.get(diff, len(bucket))
                    picked = rng.sample(bucket, min(quota, len(bucket)))
                else:
                    picked = bucket
                samples.extend(picked)

        with self._lock:
            self._pending = samples

        skipped = len(done_ids)
        print(f"Split: {split}  |  Loaded {len(samples)} samples  "
              f"(skipped {skipped} already done)")

    def get_next_task(self) -> dict | None:
        with self._lock:
            if not self._pending:
                return None
            task = self._pending.pop(0)
            task["queue"] = self
            return task

    def claim_output_id(self, category: str) -> str:
        with self._lock:
            self._seq_counters[category]     += 1
            self._category_counter[category] += 1
            self._write_distribution()
            return f"VRB_{category}_{self._seq_counters[category]:04d}"

    def _write_distribution(self):
        dist_path = OUTPUT_DIR / "category_distribution.json"
        try:
            dist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = dist_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._category_counter, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(dist_path)
        except Exception as e:
            print(f"[Warning] Failed to write distribution: {e}")

    def stats(self) -> dict:
        with self._lock:
            return {"pending": len(self._pending),
                    "categories": dict(self._category_counter)}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VRBench VQA Pipeline v2")
    parser.add_argument("--task",            type=str, default=None,
                        help="Only process this task type (maze/irregular_maze/sokoban)")
    parser.add_argument("--split",           type=str, default="eval",
                        choices=["eval", "train"])
    parser.add_argument("--train_quota",     type=str, default="240,100,60")
    parser.add_argument("--num_workers",     type=int, default=3)
    parser.add_argument("--max_samples",     type=int, default=0)
    parser.add_argument("--save_rejections", action="store_true")
    parser.add_argument("--save_generated",  action="store_true")
    args = parser.parse_args()

    tasks_filter = [args.task] if args.task else None

    global OUTPUT_DIR
    if args.split == "train":
        OUTPUT_DIR = PROJECT_ROOT / "output" / "train"

    quota_parts = [int(x) for x in args.train_quota.split(",")]
    train_quota = {"hard": quota_parts[0], "medium": quota_parts[1], "easy": quota_parts[2]}

    print("=" * 60)
    print("VRBench VQA Pipeline v2")
    print("=" * 60)
    print(f"Large model  : {MODEL_LARGE}")
    print(f"Small model  : {MODEL_SMALL} (SmallModelProbe)")
    print(f"Split        : {args.split}")
    print(f"Workers      : {args.num_workers}")
    print(f"Tasks        : {tasks_filter or TASKS}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print("=" * 60)

    queue = SampleQueue()
    queue.load_samples(tasks_filter, split=args.split, train_quota=train_quota)

    s = queue.stats()
    if s["pending"] == 0:
        print("\nNo samples to process.")
        return

    if args.max_samples > 0:
        with queue._lock:
            queue._pending = queue._pending[: args.max_samples]
        print(f"Capped to {args.max_samples} samples.")

    results         = []
    lock            = threading.Lock()
    start_time      = time.time()
    total_accepted  = 0

    print(f"\nStarting {args.num_workers} parallel workers...\n{'='*60}\n")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures: dict = {}

        def _fill():
            while len(futures) < args.num_workers:
                task = queue.get_next_task()
                if task is None:
                    break
                f = executor.submit(run_single_sample, task, args)
                futures[f] = task["sample_id"]

        _fill()

        while futures:
            done_iter = as_completed(list(futures))
            f         = next(done_iter)
            sid       = futures.pop(f)
            result    = f.result()

            with lock:
                results.append(result)
                total_accepted += result.get("accepted_count", 0)

            st       = queue.stats()
            elapsed  = time.time() - start_time
            done_n   = len(results)
            rate     = done_n / elapsed * 60 if elapsed > 0 else 0
            acc_n    = sum(1 for r in results if r["status"] == "accepted")
            rej_n    = sum(1 for r in results if r["status"] == "rejected")
            fail_n   = sum(1 for r in results if r["status"] == "failed")

            print(f"\n{'─'*60}")
            print(f"[Progress] {done_n} done  "
                  f"({acc_n}✓ {rej_n}✗ {fail_n}⚠)  "
                  f"| {total_accepted} QAs saved | {st['pending']} remaining")
            print(f"[Rate] {rate:.1f} samples/min")
            print(f"{'─'*60}\n")
            sys.stdout.flush()
            _fill()

    # Summary
    total_time = time.time() - start_time
    acc_n  = sum(1 for r in results if r["status"] == "accepted")
    rej_n  = sum(1 for r in results if r["status"] == "rejected")
    fail_n = sum(1 for r in results if r["status"] == "failed")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time         : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Samples processed  : {len(results)}")
    print(f"  Accepted (≥1 QA) : {acc_n}")
    print(f"  All rejected     : {rej_n}")
    print(f"  Failed           : {fail_n}")
    print(f"Total QAs saved    : {total_accepted}")
    print("\nCategory distribution:")
    for cat, count in sorted(queue.stats()["categories"].items()):
        if count > 0:
            bar = "█" * min(count, 40)
            print(f"  {cat:<28} {bar} {count}")
    print(f"\nOutput: {OUTPUT_DIR / 'reviewed'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

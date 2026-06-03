#!/usr/bin/env python3
"""
trajectory_gen/pipeline.py
===========================
Stage-1 privileged-trajectory generation for PF-OPSD.

For every training sample in OpenWorldQA and/or VRQABench, a
privileged teacher VLM — which can observe the ground-truth future video
v* and the correct answer y* — generates a structured concrete-reasoning
trajectory:

    d_sim  →  (p_sim  →  v̂  →  z_ver) × [1..3]  →  z_rel  →  y

The saved trajectories serve as supervised fine-tuning (SFT) data for
the student MLLM in Stage 1 of PF-OPSD training.

Output format (one JSON file per accepted trajectory)
------------------------------------------------------
{
  "sample_id":   str,
  "benchmark":   "openworldqa" | "vrqabench",
  "image_path":  str,          // anchor frame (relative to project root)
  "question":    str,
  "options":     {"A": str, "B": str, "C": str, "D": str},
  "gt_answer":   str,          // A / B / C / D
  "trajectory": {
    "d_sim":              "yes" | "no",
    "d_sim_reasoning":    str,
    "simulation_attempts": [   // [] when d_sim == "no"
      {
        "attempt_index":    int,
        "p_sim":            str,
        "rollout_frame_paths": [str, ...],  // persistent relative paths
        "z_ver":            "accept" | "reject" | "uncertain",
        "z_ver_reasoning":  str
      }, ...
    ],
    "z_rel": str,
    "y":     str                // must equal gt_answer
  },
  "trajectory_str": str,       // formatted string for SFT training target
  "_metadata": {
    "generated_at":    str,    // ISO-8601
    "teacher_model":   str,
    "world_model_type": str,
    "duration_seconds": float
  }
}

Usage
-----
    # Dry-run on 3 OpenWorldQA samples (stub world model):
    python -m trajectory_gen.pipeline \\
        --benchmark openworldqa \\
        --max_samples 3 \\
        --world_model_type stub \\
        --dry_run

    # Full run (OpenWorldQA + VRQABench, real Helios):
    python -m trajectory_gen.pipeline \\
        --benchmark all \\
        --num_workers 4 \\
        --world_model_type helios \\
        --helios_base_url $HELIOS_BASE_URL \\
        --output_dir trajectory_gen/output/trajectories
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .world_model import WorldModelClient, build_world_model

# ─────────────────────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT   = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# VRQABench data root.  Set VRB_DATA_ROOT in your environment to point to your
# local VRQABench checkout, or place it at <repo>/data/vrqabench by default.
_VRB_ROOT = Path(
    os.environ.get("VRB_DATA_ROOT", str(_REPO_ROOT / "data" / "vrqabench"))
)

# ─────────────────────────────────────────────────────────────────────────────
# Teacher VLM client
# ─────────────────────────────────────────────────────────────────────────────

class TeacherClient:
    """
    Thin wrapper around the OpenAI-compatible chat endpoint used as the
    privileged teacher.

    Parameters
    ----------
    model:    model identifier, e.g. "gemini-3.1-pro" or "gpt-5.4"
    base_url: API endpoint root
    api_key:  authentication token
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model  = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(
        self,
        prompt_text: str,
        image_paths: list[str],
        *,
        max_retries: int = 3,
        retry_sleep: float = 4.0,
    ) -> str:
        """
        Call the teacher VLM with a text prompt and zero or more images.

        Returns the raw text response.  Raises RuntimeError after
        *max_retries* failed attempts.
        """
        content: list[dict] = [{"type": "text", "text": prompt_text}]
        for path in image_paths:
            b64  = _encode_image(path)
            ext  = Path(path).suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            content.append({
                "type":      "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model    = self.model,
                    messages = [{"role": "user", "content": content}],
                )
                return resp.choices[0].message.content
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep)

        raise RuntimeError(
            f"TeacherClient: all {max_retries} attempts failed. "
            f"Last error: {last_err}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — normalised sample format
# ─────────────────────────────────────────────────────────────────────────────

def load_openworldqa_samples(split: str = "train") -> list[dict]:
    """
    Load OpenWorldQA samples from OpenWorldQA_split/{split}/*.json.

    Each returned dict has the normalised keys:
      sample_id, benchmark, image_path, future_frames,
      question, options, gt_answer, category
    """
    split_dir = _REPO_ROOT / "OpenWorldQA_split" / split
    frames_dir = _REPO_ROOT / "output" / "frames"

    samples: list[dict] = []
    for path in sorted(split_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))

        video_id    = Path(raw["video_path"]).name        # e.g. "06X2V"
        anchor_name = raw["anchor_frame"]
        if not anchor_name.endswith(".jpg"):
            anchor_name += ".jpg"

        video_frames_dir = frames_dir / video_id
        anchor_path = video_frames_dir / anchor_name

        # Future frames = all frames AFTER the anchor (temporal order)
        all_frames = sorted(
            video_frames_dir.glob("frame_*.jpg"),
            key=lambda p: p.name,
        )
        anchor_idx   = next(
            (i for i, f in enumerate(all_frames) if f.name == anchor_name), 0
        )
        future_paths = [str(f) for f in all_frames[anchor_idx + 1:]]

        samples.append({
            "sample_id":    raw["sample_id"],
            "benchmark":    "openworldqa",
            "image_path":   str(anchor_path),
            "future_frames": future_paths,
            "question":     raw["question"],
            "options":      raw["options"],
            "gt_answer":    raw["answer"],
            "category":     raw["category"],
        })

    return samples


def load_vrqabench_samples(split: str = "train") -> list[dict]:
    """
    Load VRQABench samples from VRBench-Spatial-v2/{split}/questions/*.json.

    For each sample, the future frames are extracted from the corresponding
    solution video in $VRB_DATA_ROOT/raw_data/.

    Each returned dict has the same normalised keys as load_openworldqa_samples().
    """
    questions_dir = _VRB_ROOT / "VRBench-Spatial-v2" / split / "questions"
    images_dir    = _VRB_ROOT / "VRBench-Spatial-v2" / split / "images"
    raw_data_dir  = _VRB_ROOT / "raw_data"

    samples: list[dict] = []
    for path in sorted(questions_dir.glob("VRB_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Resolve puzzle image (absolute path)
        rel_image   = raw.get("input_image", "")   # e.g. "images/maze/1/hard/images/hard_0058.png"
        image_path  = str(images_dir.parent / rel_image)

        # Resolve solution video → extract frames as future frames (v*)
        future_paths = _resolve_vrb_future_frames(
            raw.get("source_sample_id", ""),
            raw_data_dir,
        )

        sample_id = (
            path.stem                                # e.g. "VRB_C1_turn_count_0001"
            if not raw.get("source_sample_id")
            else f"VRB_{raw['source_sample_id']}"
        )

        samples.append({
            "sample_id":     sample_id,
            "benchmark":     "vrqabench",
            "image_path":    image_path,
            "future_frames": future_paths,
            "question":      raw["question"],
            "options":       raw["options"],
            "gt_answer":     raw["answer"],
            "category":      raw["category"],
            "task_type":     raw.get("task_type", ""),
        })

    return samples


def _resolve_vrb_future_frames(source_id: str, raw_data_dir: Path) -> list[str]:
    """
    Given a VRBench source_sample_id (e.g. "maze_1_hard_0058_0"), find the
    corresponding solution video and return extracted frame paths.

    Returns an empty list if the video is not found (the trajectory pipeline
    will then proceed without future frames for that sample).
    """
    if not source_id:
        return []

    # Parse: task_variant_difficulty_index_attempt
    # e.g.  maze_1_hard_0058_0  →  task=maze, variant=1, diff=hard, idx=hard_0058
    parts = source_id.split("_")
    if len(parts) < 4:
        return []

    task    = parts[0]                        # maze / sokoban / irregular_maze
    variant = parts[1]                        # 1 – 5
    diff    = parts[2]                        # hard / medium / easy
    idx_raw = "_".join(parts[3:])             # e.g. "0058_0"
    # strip trailing attempt suffix (_0) to get base video name
    base    = re.sub(r"_\d+$", "", idx_raw)   # "0058"

    video_path = (
        raw_data_dir
        / task / variant / diff
        / "videos" / f"{diff}_{base}_0.mp4"
    )
    if not video_path.exists():
        return []

    # Cache frames next to the video under a frames_cache/ sibling directory
    cache_dir = (
        raw_data_dir.parent / "output" / "frames_cache"
        / task / variant / diff / f"{diff}_{base}"
    )
    if cache_dir.exists() and list(cache_dir.glob("frame_*.jpg")):
        frames = sorted(cache_dir.glob("frame_*.jpg"), key=lambda p: p.name)
        return [str(f) for f in frames]

    # Extract with ffmpeg
    cache_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    out_pattern = str(cache_dir / "frame_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_path),
         "-vf", "fps=2", "-q:v", "2", out_pattern],
        capture_output=True,
    )
    frames = sorted(cache_dir.glob("frame_*.jpg"), key=lambda p: p.name)
    return [str(f) for f in frames]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _format_options(options: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in options.items())


def _build_phase_a_prompt(sample: dict, prev_attempts: list[dict]) -> str:
    """
    Phase A: decide d_sim + write p_sim.
    Appended to decide_simulate.txt.
    """
    base = _load_prompt("decide_simulate.txt")

    ctx = (
        f"\n\nQUESTION: {sample['question']}\n\n"
        f"OPTIONS:\n{_format_options(sample['options'])}\n\n"
        f"GROUND-TRUTH ANSWER (y*): {sample['gt_answer']}"
    )

    if prev_attempts:
        ctx += "\n\nPREVIOUS SIMULATION ATTEMPTS (all produced unusable rollouts):\n"
        for i, att in enumerate(prev_attempts, 1):
            ctx += (
                f"  Attempt {i}: p_sim = \"{att['p_sim']}\"\n"
                f"  z_ver = {att['z_ver']}: {att['z_ver_reasoning']}\n"
            )
        ctx += (
            "\nWrite a DIFFERENT p_sim that better targets the task-critical "
            "physical interaction."
        )

    return base + ctx


def _build_retry_sim_prompt(
    sample: dict,
    prev_attempts: list[dict],
) -> str:
    """
    Retry-specific Phase A prompt: d_sim is already 'yes'; teacher writes
    only a new, improved p_sim.  Uses retry_sim_prompt.txt.
    """
    base = _load_prompt("retry_sim_prompt.txt")
    ctx = (
        f"\n\nQUESTION: {sample['question']}\n\n"
        f"OPTIONS:\n{_format_options(sample['options'])}\n\n"
        f"GROUND-TRUTH ANSWER (y*): {sample['gt_answer']}\n\n"
        f"PREVIOUS SIMULATION ATTEMPTS:\n"
    )
    for i, att in enumerate(prev_attempts, 1):
        ctx += (
            f"  Attempt {i}:\n"
            f"    p_sim   : \"{att['p_sim']}\"\n"
            f"    z_ver   : {att['z_ver']}\n"
            f"    reasoning: {att['z_ver_reasoning']}\n"
        )
    return base + ctx


def _build_phase_b_prompt(
    sample: dict,
    p_sim: str,
    rollout_frame_count: int,
    future_frame_count: int,
    prev_attempts: list[dict],
) -> str:
    """
    Phase B: z_ver + z_rel + y, after seeing rollout.
    Appended to verify_rollout.txt.
    """
    base = _load_prompt("verify_rollout.txt")

    ctx = (
        f"\n\nIMAGE COUNT BREAKDOWN:\n"
        f"  Image 1:            ANCHOR FRAME\n"
        f"  Images 2 – {1 + rollout_frame_count}:  "
        f"ROLLOUT FRAMES (v̂)  [{rollout_frame_count} frames]\n"
        f"  Images {2 + rollout_frame_count} – "
        f"{1 + rollout_frame_count + future_frame_count}:  "
        f"FUTURE FRAMES (v*)  [{future_frame_count} frames]\n"
        f"\nQUESTION: {sample['question']}\n\n"
        f"OPTIONS:\n{_format_options(sample['options'])}\n\n"
        f"GROUND-TRUTH ANSWER (y*): {sample['gt_answer']}\n\n"
        f"SIMULATION PROMPT USED: \"{p_sim}\""
    )

    if prev_attempts:
        ctx += "\n\nPREVIOUS ATTEMPTS:\n"
        for i, att in enumerate(prev_attempts, 1):
            ctx += (
                f"  Attempt {i}: z_ver = {att['z_ver']}: "
                f"{att['z_ver_reasoning']}\n"
            )

    return base + ctx


def _build_abstract_prompt(sample: dict) -> str:
    """
    No-simulation path: abstract reasoning only.
    Appended to abstract_reliance.txt.
    """
    base = _load_prompt("abstract_reliance.txt")
    ctx = (
        f"\n\nQUESTION: {sample['question']}\n\n"
        f"OPTIONS:\n{_format_options(sample['options'])}\n\n"
        f"GROUND-TRUTH ANSWER (y*): {sample['gt_answer']}"
    )
    return base + ctx


# ─────────────────────────────────────────────────────────────────────────────
# Rollout frame persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_rollout_frames(
    tmp_frame_paths: list[str],
    output_dir: Path,
    sample_id: str,
    attempt_idx: int,
) -> list[str]:
    """
    Copy rollout frames from a temporary directory to a persistent cache
    and return relative paths.

    Persistent path: {output_dir}/rollouts/{sample_id}/attempt_{i}/frame_NNN.jpg
    """
    dest_dir = output_dir / "rollouts" / sample_id / f"attempt_{attempt_idx}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    for src in tmp_frame_paths:
        fname = Path(src).name
        dst   = dest_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        try:
            rel = str(dst.relative_to(_REPO_ROOT))
        except ValueError:
            rel = str(dst)
        rel_paths.append(rel)
    return rel_paths


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory string formatter  (SFT training target)
# ─────────────────────────────────────────────────────────────────────────────

# Trajectory tag constants — used identically in training and inference.
TAG_THINK_OPEN    = "<think>"
TAG_THINK_CLOSE   = "</think>"
TAG_SIM_DEC_OPEN  = "<sim_decision>"
TAG_SIM_DEC_CLOSE = "</sim_decision>"
TAG_SIM_PRM_OPEN  = "<sim_prompt>"
TAG_SIM_PRM_CLOSE = "</sim_prompt>"
TAG_VERIFY_OPEN   = "<verify>"
TAG_VERIFY_CLOSE  = "</verify>"
TAG_RELIANCE_OPEN = "<reliance>"
TAG_RELIANCE_CLOSE= "</reliance>"
TAG_ANSWER_OPEN   = "<answer>"
TAG_ANSWER_CLOSE  = "</answer>"

TRAJECTORY_TAGS = {
    "think":        (TAG_THINK_OPEN,    TAG_THINK_CLOSE),
    "sim_decision": (TAG_SIM_DEC_OPEN,  TAG_SIM_DEC_CLOSE),
    "sim_prompt":   (TAG_SIM_PRM_OPEN,  TAG_SIM_PRM_CLOSE),
    "verify":       (TAG_VERIFY_OPEN,   TAG_VERIFY_CLOSE),
    "reliance":     (TAG_RELIANCE_OPEN, TAG_RELIANCE_CLOSE),
    "answer":       (TAG_ANSWER_OPEN,   TAG_ANSWER_CLOSE),
}


def _wrap(tag: str, content: str) -> str:
    o, c = TRAJECTORY_TAGS[tag]
    return f"{o}{content}{c}"


def format_trajectory_str(trajectory: dict) -> str:
    """
    Convert a trajectory dict to the structured string that is used as the
    SFT training target (assistant turn).

    Format (d_sim = "no"):
        <think>{d_sim_reasoning}\\n{z_rel}</think>
        <sim_decision>no</sim_decision>
        <reliance>{z_rel}</reliance>
        <answer>{y}</answer>

    Format (d_sim = "yes", one accepted attempt):
        <think>{d_sim_reasoning}</think>
        <sim_decision>yes</sim_decision>
        <sim_prompt>{p_sim}</sim_prompt>
        <think>{z_ver_reasoning}</think>
        <verify>{z_ver}</verify>
        [repeat sim_prompt / think / verify for retries…]
        <reliance>{z_rel}</reliance>
        <answer>{y}</answer>

    All free-form reasoning is wrapped inside <think>…</think> tags,
    consistent with Qwen3.5's thinking-token convention.
    """
    d_sim     = trajectory.get("d_sim", "no")
    d_reason  = trajectory.get("d_sim_reasoning", "").strip()
    z_rel     = trajectory.get("z_rel", "").strip()
    y         = trajectory.get("y", "")
    attempts  = trajectory.get("simulation_attempts", [])

    parts: list[str] = []

    if d_sim != "yes" or not attempts:
        # ── No-simulation path ────────────────────────────────────────────
        think_content = d_reason
        if z_rel:
            think_content += f"\n{z_rel}"
        if think_content:
            parts.append(_wrap("think", think_content))
        parts.append(_wrap("sim_decision", "no"))
        if z_rel:
            parts.append(_wrap("reliance", z_rel))
        parts.append(_wrap("answer", y))
    else:
        # ── Simulation path ───────────────────────────────────────────────
        if d_reason:
            parts.append(_wrap("think", d_reason))
        parts.append(_wrap("sim_decision", "yes"))

        for att in attempts:
            p_sim    = att.get("p_sim", "").strip()
            z_ver    = att.get("z_ver", "uncertain")
            z_v_reas = att.get("z_ver_reasoning", "").strip()

            if p_sim:
                parts.append(_wrap("sim_prompt", p_sim))
            if z_v_reas:
                parts.append(_wrap("think", z_v_reas))
            parts.append(_wrap("verify", z_ver))

        if z_rel:
            parts.append(_wrap("reliance", z_rel))
        parts.append(_wrap("answer", y))

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict | None:
    """Extract and parse the first JSON object from a VLM response."""
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None

    if raw is None:
        # Fall back to first {...} block
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = m.group(1) if m else text.strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core trajectory generation for a single sample
# ─────────────────────────────────────────────────────────────────────────────

def generate_trajectory(
    sample: dict,
    teacher: TeacherClient,
    world_model: WorldModelClient,
    *,
    max_sim_attempts: int = 3,
    rollout_num_frames: int = 8,
    output_dir: Path | None = None,
) -> dict | None:
    """
    Generate a Stage-1 privileged trajectory for one training sample.

    Parameters
    ----------
    sample:
        Normalised sample dict (from load_openworldqa_samples or load_vrqabench_samples).
    teacher:
        Privileged teacher VLM client.
    world_model:
        Generative world model client (real Helios or stub).
    max_sim_attempts:
        Maximum number of simulation attempts before stopping (≤ 3 per paper).
    rollout_num_frames:
        Frames to extract from each generated rollout.
    output_dir:
        If provided, rollout frames are copied here for SFT persistence.
        When None, rollout frame paths reference temporary directories
        (suitable only for testing).

    Returns
    -------
    Trajectory dict (see module docstring for schema), or None if generation
    fails or the predicted answer does not match the ground-truth answer.
    """
    image_path    = sample["image_path"]
    future_frames = sample.get("future_frames", [])
    gt_answer     = sample["gt_answer"]

    # Validate anchor frame exists
    if not image_path or not Path(image_path).exists():
        return None

    phase_a_images = [image_path] + future_frames

    # ── Phase A: decide d_sim + p_sim ─────────────────────────────────────
    phase_a_prompt = _build_phase_a_prompt(sample, prev_attempts=[])
    raw_a = teacher.call(phase_a_prompt, phase_a_images)
    phase_a = _parse_json_response(raw_a)
    if not phase_a or "d_sim" not in phase_a:
        return None

    d_sim           = phase_a.get("d_sim", "no").lower().strip()
    d_sim_reasoning = phase_a.get("d_sim_reasoning", "")

    # ── No-simulation path ────────────────────────────────────────────────
    if d_sim != "yes":
        abs_prompt = _build_abstract_prompt(sample)
        raw_c      = teacher.call(abs_prompt, phase_a_images)
        phase_c    = _parse_json_response(raw_c)

        if not phase_c or "y" not in phase_c:
            return None
        if phase_c["y"] != gt_answer:
            return None   # answer-consistency filter

        traj = {
            "d_sim":               "no",
            "d_sim_reasoning":     d_sim_reasoning,
            "simulation_attempts": [],
            "z_rel":               phase_c.get("z_rel", ""),
            "y":                   phase_c["y"],
        }
        return {
            "sample_id":      sample["sample_id"],
            "benchmark":      sample["benchmark"],
            "image_path":     _make_relative(image_path),
            "question":       sample["question"],
            "options":        sample["options"],
            "gt_answer":      gt_answer,
            "trajectory":     traj,
            "trajectory_str": format_trajectory_str(traj),
        }

    # ── Simulation path ───────────────────────────────────────────────────
    p_sim        = phase_a.get("p_sim", "").strip()
    attempts: list[dict] = []
    final_phase_b: dict | None = None

    for attempt_idx in range(max_sim_attempts):
        # On retries: use the dedicated retry prompt (only asks for new p_sim;
        # does NOT re-decide d_sim, preventing the teacher from flipping to "no").
        if attempt_idx > 0:
            retry_prompt = _build_retry_sim_prompt(sample, prev_attempts=attempts)
            raw_retry    = teacher.call(retry_prompt, phase_a_images)
            phase_retry  = _parse_json_response(raw_retry)
            if not phase_retry:
                break
            p_sim = phase_retry.get("p_sim", p_sim).strip()

        if not p_sim:
            break

        # Call world model → temporary rollout frames
        try:
            tmp_rollout = world_model.generate_rollout(
                image_path, p_sim, num_frames=rollout_num_frames
            )
        except Exception:
            break

        # Persist rollout frames so SFT training can reference them
        if output_dir is not None:
            rollout_frame_paths = _save_rollout_frames(
                tmp_rollout, output_dir, sample["sample_id"], attempt_idx
            )
        else:
            rollout_frame_paths = tmp_rollout   # temp paths (testing only)

        # Phase B: verify rollout + write z_rel + predict y
        phase_b_prompt = _build_phase_b_prompt(
            sample              = sample,
            p_sim               = p_sim,
            rollout_frame_count = len(tmp_rollout),
            future_frame_count  = len(future_frames),
            prev_attempts       = attempts,
        )
        phase_b_images = [image_path] + tmp_rollout + future_frames

        raw_b   = teacher.call(phase_b_prompt, phase_b_images)
        phase_b = _parse_json_response(raw_b)
        if not phase_b:
            break

        z_ver           = phase_b.get("z_ver", "uncertain").lower().strip()
        z_ver_reasoning = phase_b.get("z_ver_reasoning", "")

        attempts.append({
            "attempt_index":      attempt_idx,
            "p_sim":              p_sim,
            "rollout_frame_paths": rollout_frame_paths,   # persistent paths
            "z_ver":              z_ver,
            "z_ver_reasoning":    z_ver_reasoning,
        })
        final_phase_b = phase_b

        if z_ver == "accept":
            break   # accepted rollout — stop retrying

    if not final_phase_b or "y" not in final_phase_b:
        return None
    if final_phase_b["y"] != gt_answer:
        return None   # answer-consistency filter

    traj = {
        "d_sim":               "yes",
        "d_sim_reasoning":     d_sim_reasoning,
        "simulation_attempts": attempts,
        "z_rel":               final_phase_b.get("z_rel", ""),
        "y":                   final_phase_b["y"],
    }
    return {
        "sample_id":      sample["sample_id"],
        "benchmark":      sample["benchmark"],
        "image_path":     _make_relative(image_path),
        "question":       sample["question"],
        "options":        sample["options"],
        "gt_answer":      gt_answer,
        "trajectory":     traj,
        "trajectory_str": format_trajectory_str(traj),
    }


def _make_relative(abs_path: str) -> str:
    """Convert an absolute path to a path relative to the repo root."""
    try:
        return str(Path(abs_path).relative_to(_REPO_ROOT))
    except ValueError:
        try:
            return str(Path(abs_path).relative_to(_VRB_ROOT))
        except ValueError:
            return abs_path


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryGenerator — orchestrates batches with resume, concurrency, logging
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryGenerator:
    """
    Orchestrates trajectory generation for a list of training samples.

    Parameters
    ----------
    teacher:       Privileged teacher VLM.
    world_model:   Generative world model (Helios or stub).
    output_dir:    Directory where trajectory JSON files are saved.
    max_sim_attempts:  Max simulation retries per sample (paper uses 3).
    rollout_num_frames: Frames to extract per rollout.
    num_workers:   Thread-pool workers for parallelism.
    """

    def __init__(
        self,
        teacher:             TeacherClient,
        world_model:         WorldModelClient,
        output_dir:          str | Path,
        max_sim_attempts:    int   = 3,
        rollout_num_frames:  int   = 8,
        num_workers:         int   = 4,
    ):
        self.teacher            = teacher
        self.world_model        = world_model
        self.output_dir         = Path(output_dir)
        self.max_sim_attempts   = max_sim_attempts
        self.rollout_num_frames = rollout_num_frames
        self.num_workers        = num_workers

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run(self, samples: list[dict]) -> dict:
        """
        Generate trajectories for all samples (parallel).

        Returns a summary dict with counts and per-benchmark breakdown.
        """
        # Filter already-done samples (resume support)
        pending = [s for s in samples if not self._is_done(s["sample_id"])]
        print(
            f"[TrajectoryGen] {len(pending)} pending / {len(samples)} total  "
            f"(output: {self.output_dir})"
        )

        stats = defaultdict(lambda: {"accepted": 0, "rejected": 0, "error": 0})
        done  = [0]

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(self._run_one, s): s["sample_id"]
                for s in pending
            }
            for fut in as_completed(futures):
                sample_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"status": "error", "reason": str(exc),
                              "benchmark": "unknown"}

                benchmark = result.get("benchmark", "unknown")
                with self._lock:
                    stats[benchmark][result["status"]] += 1
                    done[0] += 1
                    if done[0] % 20 == 0 or done[0] == len(pending):
                        self._print_progress(done[0], len(pending), stats)

        return {b: dict(v) for b, v in stats.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self, sample: dict) -> dict:
        start = time.time()
        sid   = sample["sample_id"]

        try:
            traj = generate_trajectory(
                sample,
                self.teacher,
                self.world_model,
                max_sim_attempts   = self.max_sim_attempts,
                rollout_num_frames = self.rollout_num_frames,
                output_dir         = self.output_dir,   # for rollout persistence
            )
        except Exception as exc:
            print(f"  [ERROR] {sid}: {exc}")
            return {"status": "error", "sample_id": sid,
                    "benchmark": sample.get("benchmark", "unknown")}

        if traj is None:
            print(f"  [REJECT] {sid}: answer-inconsistent or parse failure")
            return {"status": "rejected", "sample_id": sid,
                    "benchmark": sample.get("benchmark", "unknown")}

        # Attach metadata and save
        traj["_metadata"] = {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "teacher_model":    self.teacher.model,
            "world_model_type": type(self.world_model).__name__,
            "duration_seconds": round(time.time() - start, 2),
        }
        out_path = self.output_dir / f"{sid}.json"
        out_path.write_text(
            json.dumps(traj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"  [OK] {sid}  d_sim={traj['trajectory']['d_sim']}"
            f"  attempts={len(traj['trajectory']['simulation_attempts'])}"
            f"  {traj['_metadata']['duration_seconds']:.1f}s"
        )
        return {"status": "accepted", "sample_id": sid,
                "benchmark": sample.get("benchmark", "unknown")}

    def _is_done(self, sample_id: str) -> bool:
        return (self.output_dir / f"{sample_id}.json").exists()

    @staticmethod
    def _print_progress(
        done: int,
        total: int,
        stats: dict,
    ) -> None:
        parts = []
        for bench, cnts in sorted(stats.items()):
            parts.append(
                f"{bench}: {cnts['accepted']}ok "
                f"{cnts['rejected']}rej {cnts['error']}err"
            )
        print(f"  [{done:>5}/{total}]  " + "  |  ".join(parts))


# ─────────────────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Stage-1 privileged trajectories for PF-OPSD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    g_data = p.add_argument_group("Data")
    g_data.add_argument(
        "--benchmark",
        default="all",
        choices=["openworldqa", "vrqabench", "all"],
        help="Which benchmark(s) to generate trajectories for.",
    )
    g_data.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Process at most this many samples (0 = all).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    g_out = p.add_argument_group("Output")
    g_out.add_argument(
        "--output_dir",
        default="trajectory_gen/output/trajectories",
        help="Directory to write trajectory JSON files.",
    )

    # ── Teacher VLM ───────────────────────────────────────────────────────
    g_teacher = p.add_argument_group("Teacher VLM")
    g_teacher.add_argument(
        "--teacher_model",
        default="gemini-3.1-pro",
        help="Teacher model identifier.",
    )
    g_teacher.add_argument(
        "--teacher_base_url",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API endpoint for the teacher.",
    )
    g_teacher.add_argument(
        "--teacher_api_key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for the teacher endpoint ($OPENAI_API_KEY by default).",
    )

    # ── World model ───────────────────────────────────────────────────────
    g_wm = p.add_argument_group("World Model")
    g_wm.add_argument(
        "--world_model_type",
        default="stub",
        choices=["stub", "helios"],
        help="'stub' for offline debugging; 'helios' for real rollouts.",
    )
    g_wm.add_argument(
        "--helios_base_url",
        default=os.environ.get("HELIOS_BASE_URL", ""),
        help="Helios HTTP endpoint (required when --world_model_type=helios).",
    )
    g_wm.add_argument(
        "--helios_model",
        default="helios-vrbench",
        help="Helios model variant to use.",
    )
    g_wm.add_argument(
        "--rollout_num_frames",
        type=int,
        default=8,
        help="Number of frames to extract from each generated rollout.",
    )
    g_wm.add_argument(
        "--max_sim_attempts",
        type=int,
        default=3,
        help="Maximum simulation retries per sample (paper uses 3).",
    )

    # ── Runtime ───────────────────────────────────────────────────────────
    g_rt = p.add_argument_group("Runtime")
    g_rt.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker threads.",
    )
    g_rt.add_argument(
        "--dry_run",
        action="store_true",
        help="Print sample counts and exit without calling any APIs.",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load samples ──────────────────────────────────────────────────────
    samples: list[dict] = []
    if args.benchmark in ("openworldqa", "all"):
        ow = load_openworldqa_samples("train")
        print(f"[Data] Loaded {len(ow)} OpenWorldQA train samples.")
        samples.extend(ow)
    if args.benchmark in ("vrqabench", "all"):
        vrb = load_vrqabench_samples("train")
        print(f"[Data] Loaded {len(vrb)} VRQABench train samples.")
        samples.extend(vrb)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        print(f"[Data] Capped to {len(samples)} samples (--max_samples).")

    if args.dry_run:
        print(f"[DryRun] Would process {len(samples)} samples. Exiting.")
        return

    # ── Build teacher + world model ───────────────────────────────────────
    teacher = TeacherClient(
        model    = args.teacher_model,
        base_url = args.teacher_base_url,
        api_key  = args.teacher_api_key,
    )

    wm_config: dict = {"type": args.world_model_type}
    if args.world_model_type == "helios":
        if not args.helios_base_url:
            print(
                "[Error] --helios_base_url is required when "
                "--world_model_type=helios",
                file=sys.stderr,
            )
            sys.exit(1)
        wm_config.update({
            "base_url": args.helios_base_url,
            "model":    args.helios_model,
        })
    world_model = build_world_model(wm_config)

    # ── Run ───────────────────────────────────────────────────────────────
    generator = TrajectoryGenerator(
        teacher            = teacher,
        world_model        = world_model,
        output_dir         = args.output_dir,
        max_sim_attempts   = args.max_sim_attempts,
        rollout_num_frames = args.rollout_num_frames,
        num_workers        = args.num_workers,
    )
    summary = generator.run(samples)

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Trajectory Generation Summary")
    print("=" * 55)
    total_ok = total_rej = total_err = 0
    for bench, cnts in sorted(summary.items()):
        ok, rej, err = cnts["accepted"], cnts["rejected"], cnts["error"]
        total_ok += ok; total_rej += rej; total_err += err
        print(
            f"  {bench:<12}  accepted={ok}  rejected={rej}  errors={err}"
        )
    print(f"  {'TOTAL':<12}  accepted={total_ok}  rejected={total_rej}"
          f"  errors={total_err}")
    print(f"  Saved to: {Path(args.output_dir).resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
stage2_pfopsd.py
================
Stage 2 of PF-OPSD: Privileged-Future On-Policy Self-Distillation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPSD ARCHITECTURE — SAME MODEL, DIFFERENT INPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PF-OPSD is On-Policy Self-Distillation.  "Self" means the teacher and
student are THE SAME model weights pi_theta:

  Student (no privilege):
      input  = (anchor, rollout_v̂, question, options)
      output = trajectory decisions (d_sim, p_sim, z_ver, z_rel, y)

  Teacher (privileged):
      input  = (anchor, rollout_v̂, future_v*, question, options, y*)
      output = SAME decisions, but with privileged context
      -> produces P_t+(a) = pi_theta(a | c_t+ = c_t union {v*, y*}, h_t^s)

The distillation target q*_t uses P_t+(a) to weight the advantage:
      q*_t(a) ∝ P_t+(a) · exp(A_t+(a) / tau_A)     [paper Eq. 6]

P_t+(a) is computed by get_student_priv_probs() which feeds the SAME
student model with the extra privileged frames (v*) and extracts token
log-probabilities for each candidate action.

Qwen3.6-27B (the 27B LLM) is used ONLY for rollout quality assessment
(compare v̂ with v* to derive false_accept / false_reject labels for
the reward R+(τ_a)).  It does NOT provide P_t+(a).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each training step:
  1. Sample a mini-batch of (sample, future_frames) pairs.
  2. Generate on-policy trajectories from the current student policy.
  3. For each discrete node (d_sim, z_ver, y):
       a. Force each candidate action, greedily complete τ_a.
       b. Score via E+ (Qwen3.6-27B) -> R+(τ_a) = Q_t+(a).
       c. Run student WITH v* -> P_t+(a) = pi_theta(a | c_t+, h_t^s).
       d. Compute advantage A_t+(a) = Q_t+(a) - V_t+,
          where V_t+ = Σ_a P_t+(a) · Q_t+(a).
       e. Build target q*_t(a) ∝ P_t+(a) · exp(A_t+(a) / tau_A).
       f. L_disc = KL(sg[q*_t] || pi_theta(· | c_t, h_t^s)).
  4. For each text node (p_sim, z_rel):
       a. Sample K candidates from student (no privilege).
       b. Score each via E+ -> R+(τ_a_k).
       c. Advantages A_t+(a_k) = Q_t+(a_k) - mean(Q).
       d. L_text = -Σ_k w_k · log pi_theta(a_k | c_t, h_t^s).
  5. Back-propagate L = L_SFT + L_disc + L_text + λ_call·N_sim.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper hyperparameters (Appendix A):
    100 on-policy update steps
    K=4 text-node candidates, tau_A=0.5
    (λ_sim, λ_FA, λ_FR) = (0.05, 0.50, 0.25),  λ_call=0.02
    lr=5e-6, AdamW, bfloat16, gradient-checkpointing

Usage
-----
    python -m train.stage2_pfopsd --config train/configs/pfopsd.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from accelerate.utils import set_seed
from peft import PeftConfig, PeftModel
from transformers import (
    AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments,
)

from .dataset import OnPolicyDataset, _SYSTEM_PROMPT
from .losses import pfopsd_loss, PFOPSDLossComponents
from .reward import PrivilegedEvaluator, compute_reward, TrajectoryReward

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _default_cfg() -> dict:
    return {
        # Model — start from Stage 1 checkpoint
        "model_name_or_path":    "checkpoints/stage1_sft/final",
        "attn_implementation":   "flash_attention_2",
        # Data
        "openworldqa_split_dir":   "OpenWorldQA_split/train",
        "vrb_split_dir":         "",        # leave empty to skip VRQABench
        "openworldqa_frames_base": "output/frames",
        # Output
        "output_dir":            "checkpoints/stage2_pfopsd",
        # Training schedule (paper: 100 on-policy update steps)
        "num_update_steps":      100,
        "batch_size":            4,         # samples per update step
        "gradient_accumulation_steps": 8,
        "max_length":            16384,
        "learning_rate":         5e-6,
        "warmup_ratio":          0.05,
        "max_grad_norm":         1.0,
        "seed":                  42,
        "save_steps":            20,
        "log_steps":             5,
        # PF-OPSD hyperparameters (paper Table A.1)
        "advantage_temperature": 0.5,
        "lambda_sim":            0.05,
        "lambda_fa":             0.50,
        "lambda_fr":             0.25,
        "lambda_call":           0.02,
        "k_text_candidates":     4,
        "text_sample_temperature": 1.1,
        "max_sim_attempts":      3,
        "rollout_num_frames":    8,
        # Privileged evaluator
        "evaluator_model":       "Qwen3.6-27B",
        "evaluator_base_url":    os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "evaluator_api_key":     "",        # falls back to $OPENAI_API_KEY
        # World model (for on-policy rollout generation)
        "world_model_type":      "stub",    # "stub" | "helios"
        "helios_base_url":       "",
        "helios_model":          "helios-vrbench",
        # SFT regularisation (paper: L = L_SFT + L_disc + L_text + λ_call·N_sim)
        # Point to the same trajectory dir used in Stage 1.
        # Set lambda_sft=0 to disable SFT regularisation entirely.
        "sft_trajectory_dir":    "trajectory_gen/output/trajectories",
        "sft_batch_size":        2,     # SFT mini-batch per on-policy step
        "lambda_sft":            1.0,
    }


def _load_cfg(config_path: str | None) -> dict:
    cfg = _default_cfg()
    if config_path and Path(config_path).exists():
        with open(config_path) as fh:
            override = yaml.safe_load(fh) or {}
        cfg.update(override)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory generation helpers
# ─────────────────────────────────────────────────────────────────────────────

DISC_ACTIONS = {
    "d_sim": ["yes", "no"],
    "z_ver": ["accept", "reject", "uncertain"],
    "y":     ["A", "B", "C", "D"],
}


def _generate_trajectory_on_policy(
    model,
    processor,
    sample: dict,
    world_model,
    cfg: dict,
    device,
) -> dict | None:
    """
    Generate one trajectory by decoding from the current student policy.
    Uses greedy decoding for discrete nodes and sampling for text nodes.

    Returns a trajectory dict or None on failure.
    """
    from trajectory_gen.pipeline import (
        format_trajectory_str,
        _build_phase_a_prompt,
        _build_phase_b_prompt,
        _build_abstract_prompt,
        _parse_json_response,
    )
    from trajectory_gen.world_model import WorldModelClient
    from PIL import Image

    image_path    = sample["image_path"]
    future_frames = sample.get("future_frames", [])
    if not Path(image_path).exists():
        return None

    def _vlm_call(prompt_text: str, img_paths: list[str], greedy: bool = True) -> str:
        images = [Image.open(p).convert("RGB") for p in img_paths if Path(p).exists()]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": prompt_text}]
                ),
            },
        ]
        enc = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens":  512,
            "do_sample":       not greedy,
            "temperature":     cfg["text_sample_temperature"] if not greedy else 1.0,
        }
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    phase_a_images = [image_path] + future_frames
    raw_a    = _vlm_call(
        _build_phase_a_prompt(sample, prev_attempts=[]),
        phase_a_images,
        greedy=True,
    )
    phase_a  = _parse_json_response(raw_a)
    if not phase_a:
        return None

    d_sim           = phase_a.get("d_sim", "no").lower().strip()
    d_sim_reasoning = phase_a.get("d_sim_reasoning", "")

    if d_sim != "yes":
        raw_c   = _vlm_call(_build_abstract_prompt(sample), phase_a_images, greedy=False)
        phase_c = _parse_json_response(raw_c)
        if not phase_c:
            return None
        return {
            "d_sim": "no", "d_sim_reasoning": d_sim_reasoning,
            "simulation_attempts": [],
            "z_rel": phase_c.get("z_rel", ""), "y": phase_c.get("y", ""),
        }

    p_sim    = phase_a.get("p_sim", "").strip()
    attempts = []
    final_b  = None

    for attempt_idx in range(cfg["max_sim_attempts"]):
        if not p_sim:
            break
        try:
            rollout_tmp = world_model.generate_rollout(
                image_path, p_sim,
                num_frames=cfg["rollout_num_frames"],
            )
        except Exception:
            break

        phase_b_prompt = _build_phase_b_prompt(
            sample=sample, p_sim=p_sim,
            rollout_frame_count=len(rollout_tmp),
            future_frame_count=len(future_frames),
            prev_attempts=attempts,
        )
        raw_b   = _vlm_call(
            phase_b_prompt,
            [image_path] + rollout_tmp + future_frames,
            greedy=False,
        )
        phase_b = _parse_json_response(raw_b)
        if not phase_b:
            break

        z_ver = phase_b.get("z_ver", "uncertain").lower().strip()
        attempts.append({
            "attempt_index":      attempt_idx,
            "p_sim":              p_sim,
            "rollout_frame_paths": rollout_tmp,   # temp for scoring only
            "z_ver":              z_ver,
            "z_ver_reasoning":    phase_b.get("z_ver_reasoning", ""),
        })
        final_b = phase_b
        if z_ver == "accept":
            break

        # retry p_sim
        from trajectory_gen.pipeline import _build_retry_sim_prompt, _parse_json_response as _pjr
        raw_r   = _vlm_call(_build_retry_sim_prompt(sample, attempts), phase_a_images)
        phase_r = _pjr(raw_r)
        if phase_r:
            p_sim = phase_r.get("p_sim", p_sim).strip()

    if not final_b:
        return None
    return {
        "d_sim": "yes", "d_sim_reasoning": d_sim_reasoning,
        "simulation_attempts": attempts,
        "z_rel": final_b.get("z_rel", ""), "y": final_b.get("y", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy-completion helper
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_complete(
    model, processor, sample: dict, world_model,
    partial_traj: dict, start_after: str,
    cfg: dict, device,
) -> dict:
    """
    Given a partial trajectory (with some nodes already forced), greedily
    complete the remaining nodes from the current student policy.

    start_after: which node was just forced.  Completion starts from the
                 NEXT node in the canonical order:
                   d_sim -> p_sim -> z_ver -> z_rel -> y

    Returns a complete trajectory dict.
    """
    from trajectory_gen.pipeline import (
        _build_phase_b_prompt, _build_abstract_prompt,
        _build_retry_sim_prompt, _parse_json_response,
    )
    from PIL import Image

    image_path    = sample["image_path"]
    future_frames = sample.get("future_frames", [])

    def _call(prompt, img_paths, greedy=True):
        imgs = [Image.open(p).convert("RGB") for p in img_paths if Path(p).exists()]
        msgs = [
            {"role": "system",    "content": _SYSTEM_PROMPT},
            {"role": "user",      "content": (
                [{"type": "image", "image": img} for img in imgs]
                + [{"type": "text", "text": prompt}]
            )},
        ]
        enc = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=256, do_sample=not greedy,
                temperature=cfg.get("text_sample_temperature", 1.1) if not greedy else 1.0,
            )
        return processor.tokenizer.decode(
            out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

    traj = dict(partial_traj)

    # ── If d_sim was just forced to "yes" and no attempts yet -> generate them
    if start_after == "d_sim" and traj.get("d_sim") == "yes" \
            and not traj.get("simulation_attempts"):
        p_sim = traj.get("_forced_p_sim", "")
        attempts = []
        final_b  = None
        for attempt_idx in range(cfg["max_sim_attempts"]):
            if not p_sim:
                # Ask student to write a p_sim
                from trajectory_gen.pipeline import _build_phase_a_prompt
                raw = _call(_build_phase_a_prompt(sample, []), [image_path] + future_frames)
                pa  = _parse_json_response(raw)
                if pa:
                    p_sim = pa.get("p_sim", "").strip()
            if not p_sim:
                break
            try:
                rollout = world_model.generate_rollout(
                    image_path, p_sim, num_frames=cfg["rollout_num_frames"]
                )
            except Exception:
                break
            phase_b_prompt = _build_phase_b_prompt(
                sample=sample, p_sim=p_sim,
                rollout_frame_count=len(rollout),
                future_frame_count=len(future_frames),
                prev_attempts=attempts,
            )
            raw_b   = _call(phase_b_prompt, [image_path] + rollout + future_frames)
            phase_b = _parse_json_response(raw_b)
            if not phase_b:
                break
            z_ver = phase_b.get("z_ver", "uncertain").lower().strip()
            attempts.append({
                "attempt_index": attempt_idx,
                "p_sim": p_sim,
                "rollout_frame_paths": rollout,
                "z_ver": z_ver,
                "z_ver_reasoning": phase_b.get("z_ver_reasoning", ""),
            })
            final_b = phase_b
            if z_ver == "accept":
                break
            from trajectory_gen.pipeline import _build_retry_sim_prompt
            raw_r   = _call(_build_retry_sim_prompt(sample, attempts), [image_path] + future_frames)
            phase_r = _parse_json_response(raw_r)
            if phase_r:
                p_sim = phase_r.get("p_sim", p_sim).strip()
        traj["simulation_attempts"] = attempts
        if final_b:
            traj.setdefault("z_rel", final_b.get("z_rel", ""))
            traj.setdefault("y",     final_b.get("y", ""))
        start_after = "z_rel"   # z_rel and y already set from final_b

    # ── If d_sim was just forced to "no" -> abstract reasoning path
    if start_after == "d_sim" and traj.get("d_sim") == "no":
        raw_c   = _call(_build_abstract_prompt(sample), [image_path] + future_frames)
        phase_c = _parse_json_response(raw_c)
        if phase_c:
            traj["z_rel"] = phase_c.get("z_rel", "")
            traj["y"]     = phase_c.get("y", "")
        return traj

    # ── If z_ver was just forced -> need z_rel and y
    if start_after in ("z_ver", "p_sim"):
        attempts = traj.get("simulation_attempts", [])
        last_att = attempts[-1] if attempts else {}
        rollout  = last_att.get("rollout_frame_paths", [])
        from trajectory_gen.pipeline import _build_phase_b_prompt
        phase_b_prompt = _build_phase_b_prompt(
            sample=sample,
            p_sim=last_att.get("p_sim", ""),
            rollout_frame_count=len(rollout),
            future_frame_count=len(future_frames),
            prev_attempts=attempts[:-1],
        )
        raw_b   = _call(phase_b_prompt, [image_path] + rollout + future_frames)
        phase_b = _parse_json_response(raw_b)
        if phase_b:
            traj.setdefault("z_rel", phase_b.get("z_rel", ""))
            traj.setdefault("y",     phase_b.get("y", ""))
        return traj

    # ── If z_rel was just forced -> only y remains
    if start_after == "z_rel":
        if "y" not in traj or not traj["y"]:
            # Ask student for final answer given current context
            from trajectory_gen.pipeline import TAG_ANSWER_OPEN
            prefix = traj.get("z_rel", "")
            raw_y  = _call(f"Given your analysis:\n{prefix}\nOutput the answer A/B/C/D as JSON: {{\"y\":\"...\"}}",
                           [image_path])
            py = _parse_json_response(raw_y)
            if py:
                traj["y"] = py.get("y", "A")
        return traj

    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Node scoring — force each candidate, greedy-complete, score
# ─────────────────────────────────────────────────────────────────────────────

def _score_discrete_node(
    model, processor, sample, world_model,
    base_trajectory: dict, node_name: str,
    evaluator: "PrivilegedEvaluator", cfg: dict, device,
) -> tuple[list[float], list[str], Optional[dict[str, float]]]:
    """
    For each candidate action at a discrete node:
      1. Force the action.
      2. GREEDY-COMPLETE the remaining trajectory with the current student.
      3. Score via privileged evaluator -> Q_t+(a).
    Also call the evaluator for P_t+(a) = pi+(a | c_t+, h_t^s).

    Returns (q_values, candidates, priv_probs_dict)
      q_values:    list[float]   Q_t+(a) for each candidate
      candidates:  list[str]
      priv_probs:  dict {action: prob} or None
    """
    candidates = DISC_ACTIONS.get(node_name, [])
    q_values   = []
    future_paths = sample.get("future_frames", [])
    anchor_path  = sample["image_path"]

    for action in candidates:
        # Build partial trajectory with the forced action
        partial = dict(base_trajectory)

        if node_name == "d_sim":
            partial["d_sim"] = action
            partial.pop("simulation_attempts", None)
            partial.pop("z_rel", None)
            partial.pop("y", None)

        elif node_name == "y":
            partial["y"] = action
            # y is the last node — no further completion needed

        elif node_name == "z_ver":
            atts = list(partial.get("simulation_attempts", []))
            if atts:
                atts[-1] = dict(atts[-1], z_ver=action)
            partial["simulation_attempts"] = atts
            partial.pop("z_rel", None)
            partial.pop("y", None)

        # Greedy-complete the remaining trajectory
        completed = _greedy_complete(
            model, processor, sample, world_model,
            partial, start_after=node_name, cfg=cfg, device=device,
        )

        reward = compute_reward(
            trajectory   = completed,
            sample       = sample,
            future_paths = future_paths,
            evaluator    = evaluator,
            repo_root    = _REPO_ROOT,
            lambda_sim   = cfg["lambda_sim"],
            lambda_fa    = cfg["lambda_fa"],
            lambda_fr    = cfg["lambda_fr"],
        )
        q_values.append(reward.total)

    # ── Get P_t+(a) from the STUDENT MODEL with privileged context  ──────────
    # This is the "self" in Self-Distillation:
    #   P_t+(a) = pi_theta(a | c_t+ = c_t union {v*, y*}, h_t^s)
    # Same model weights as the student, but with v* and y* added to input.
    # NOT from Qwen3.6-27B — that is only used for rollout quality scoring.
    last_att     = (base_trajectory.get("simulation_attempts") or [{}])[-1]
    rollout_abs  = [f for f in last_att.get("rollout_frame_paths", [])
                    if Path(f).exists()]
    prefix_str   = _build_prefix(base_trajectory, node_name, sample)

    priv_probs_list = get_student_priv_probs(
        model         = model,
        processor     = processor,
        sample        = sample,
        future_paths  = sample.get("future_frames", []),
        rollout_paths = rollout_abs,
        node_name     = node_name,
        candidates    = candidates,
        prefix_str    = prefix_str,
        device        = device,
    )

    return q_values, candidates, priv_probs_list


# ─────────────────────────────────────────────────────────────────────────────
# P_t+(a) — student model with privileged context
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_student_priv_probs(
    model,
    processor,
    sample:       dict,
    future_paths: list[str],
    rollout_paths: list[str],
    node_name:    str,
    candidates:   list[str],
    prefix_str:   str,
    device,
) -> list[float] | None:
    """
    Compute P_t+(a) = pi_theta(a | c_t+, h_t^s) for each candidate action.

    This is THE core of OPSD "self"-distillation:
      - c_t  = (anchor, rollout_v̂, question, options)          <- student view
      - c_t+ = (anchor, rollout_v̂, future_v*, question, y*)   <- privileged view
    Both use THE SAME model weights pi_theta.  The difference is only the input.

    We feed c_t+ through the model together with the current trajectory
    prefix h_t^s, then extract the next-token log-probability for each
    candidate action string.

    Parameters
    ----------
    sample:        training sample with image_path, question, options, gt_answer
    future_paths:  absolute paths to future frames (v*) — privileged input
    rollout_paths: rollout frames from the last simulation attempt (v̂)
    node_name:     "d_sim" | "z_ver" | "y"
    candidates:    candidate action strings (e.g. ["yes", "no"])
    prefix_str:    trajectory text prefix up to this node (from format_trajectory_str)
    device:        torch device

    Returns
    -------
    Normalised list of probabilities [p_a1, p_a2, ...], same order as candidates.
    Returns None on failure (caller falls back to uniform).
    """
    from PIL import Image

    anchor_path = sample["image_path"]

    # Build privileged image list: anchor + rollout + future (v*)
    priv_img_paths = [anchor_path] + rollout_paths + future_paths
    images = []
    for p in priv_img_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            pass

    if not images:
        return None

    # System prompt — same as student's, but note privileged context
    _PRIV_SYSTEM = (
        "You are an AI assistant with access to the ground-truth future outcome. "
        "The images include the anchor frame, world-model rollout frames, and "
        "the actual future frames (v*). Use all of this information to make the "
        "best possible decision."
    )

    # Build the question text with y* visible (privileged)
    opts_text = "\n".join(f"  {k}: {v}" for k, v in sample["options"].items())
    user_text = (
        f"Question: {sample['question']}\n\nOptions:\n{opts_text}\n\n"
        f"[PRIVILEGED] Correct answer: {sample['gt_answer']}"
    )

    messages = [
        {"role": "system", "content": _PRIV_SYSTEM},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": user_text}]
            ),
        },
    ]

    # Add the trajectory prefix as the beginning of the assistant turn
    if prefix_str.strip():
        messages.append({"role": "assistant", "content": prefix_str.strip()})

    try:
        enc = processor.apply_chat_template(
            messages,
            tokenize              = True,
            add_generation_prompt = bool(not prefix_str.strip()),
            return_dict           = True,
            return_tensors        = "pt",
            padding               = False,
            truncation            = True,
            max_length            = 4096,
        ).to(device)

        with torch.no_grad():
            out = model(**enc)

        # last-token logits
        logits     = out.logits[0, -1, :]     # (vocab_size,)
        log_probs  = torch.log_softmax(logits, dim=-1)

        # For each candidate action string, get the log-prob of its FIRST token
        tok = processor.tokenizer
        cand_lps: list[float] = []
        for cand in candidates:
            # tokenise candidate (no special tokens)
            cand_ids = tok.encode(" " + cand, add_special_tokens=False)
            if not cand_ids:
                cand_ids = tok.encode(cand, add_special_tokens=False)
            if not cand_ids:
                cand_lps.append(-1e9)
                continue
            cand_lps.append(log_probs[cand_ids[0]].item())

        # Normalise to probabilities
        max_lp = max(cand_lps)
        probs  = [float(torch.exp(torch.tensor(lp - max_lp))) for lp in cand_lps]
        total  = sum(probs)
        if total <= 0:
            return None
        return [p / total for p in probs]

    except Exception:
        return None




class PFOPSDTrainer(Trainer):
    """
    Privileged-Future On-Policy Self-Distillation trainer.

    Built on the HuggingFace / trl ``Trainer`` base (trl's own trainers are
    thin subclasses of the same class).  The optimizer, LR scheduler, gradient
    accumulation, distributed/bf16 handling, checkpointing and logging are all
    delegated to the base ``Trainer``.  Only the *loss computation* is custom:
    for every on-policy sample we

      1. roll out a trajectory from the current student policy,
      2. score the discrete (d_sim, z_ver, y) and text (p_sim, z_rel) nodes via
         the privileged evaluator E+ and the student's privileged forward pass,
      3. assemble ``pfopsd_loss`` (L_disc + L_text + λ_call·N_sim), and
      4. add a λ_sft·L_SFT regularisation term drawn from the Stage-1
         trajectory data (paper Eq. 9).

    On-policy "examples" are raw sample dicts; ``_prepare_inputs`` therefore
    passes them straight through instead of moving tensors to device.
    """

    def __init__(
        self,
        *args,
        pf_cfg: dict,
        world_model,
        evaluator: "PrivilegedEvaluator",
        sft_loader_iter=None,
        lambda_sft: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pf_cfg          = pf_cfg
        self.world_model     = world_model
        self.evaluator       = evaluator
        self.sft_loader_iter = sft_loader_iter
        self.lambda_sft      = lambda_sft

    # On-policy inputs are a list of raw dicts (strings / paths) — keep as-is.
    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from .losses import compute_advantages

        cfg     = self.pf_cfg
        device  = self.args.device
        # Generation / no-grad scoring run on the unwrapped module so that
        # ``model.generate`` works under DDP; the L_SFT forward keeps the
        # wrapped model so its gradients flow.
        gen_model = self.accelerator.unwrap_model(model)
        processor = self.processing_class

        step_disc, step_text, step_pen, step_rewards = [], [], [], []

        for sample in inputs:
            # 1. Generate on-policy trajectory from current student policy
            traj = _generate_trajectory_on_policy(
                gen_model, processor, sample, self.world_model, cfg, device
            )
            if traj is None:
                continue

            # 2. Score discrete nodes (d_sim, z_ver, y)
            disc_nodes: list[dict] = []
            for node_name in ("d_sim", "z_ver", "y"):
                q_vals, cands, priv_probs_list = _score_discrete_node(
                    gen_model, processor, sample, self.world_model,
                    traj, node_name, self.evaluator, cfg, device,
                )
                if not q_vals:
                    continue
                advs = compute_advantages(
                    q_vals, priv_probs=priv_probs_list, node_type="disc",
                )
                # Student-side logits π_θ(a|c_t) must be differentiable — use the
                # WRAPPED model so gradients flow (and DDP syncs them).
                logits = _score_candidate_logits(
                    model, processor, sample, traj, node_name, cands, device
                )
                disc_nodes.append({
                    "student_logits": logits,
                    "advantages":     advs,
                    "priv_probs":     priv_probs_list,
                })

            # 3. Score text nodes (p_sim, z_rel)
            text_nodes: list[dict] = []
            for node_name in ("p_sim", "z_rel"):
                # Candidate sampling runs on the unwrapped model (model.generate);
                q_vals_text, cand_texts = _score_text_node(
                    gen_model, processor, sample, traj, node_name,
                    self.evaluator, cfg, device,
                )
                if not q_vals_text:
                    continue
                advs = compute_advantages(q_vals_text, node_type="text")
                # log π_θ(a_k|c_t) is recomputed WITH gradient on the wrapped model.
                prefix = _build_prefix(traj, node_name, sample)
                cand_log_probs = torch.stack([
                    _score_string_lp(model, processor, prefix, t, device)
                    for t in cand_texts
                ])
                text_nodes.append({
                    "candidate_log_probs": cand_log_probs,
                    "advantages":          advs,
                })

            # 4. PF-OPSD distillation loss for this sample
            n_sim  = len(traj.get("simulation_attempts", []))
            losses = pfopsd_loss(
                disc_nodes  = disc_nodes,
                text_nodes  = text_nodes,
                n_sim       = n_sim,
                temperature = cfg["advantage_temperature"],
                lambda_call = cfg["lambda_call"],
            )
            step_disc.append(losses.l_disc)
            step_text.append(losses.l_text)
            step_pen.append(losses.l_sim_pen)

            reward = compute_reward(
                trajectory   = traj,
                sample       = sample,
                future_paths = sample.get("future_frames", []),
                evaluator    = self.evaluator,
                repo_root    = _REPO_ROOT,
                lambda_sim   = cfg["lambda_sim"],
                lambda_fa    = cfg["lambda_fa"],
                lambda_fr    = cfg["lambda_fr"],
            )
            step_rewards.append(reward.total)

        # ── Aggregate PF-OPSD distillation losses ─────────────────────────
        l_d  = torch.stack(step_disc).mean() if step_disc else torch.zeros((), device=device)
        l_t  = torch.stack(step_text).mean() if step_text else torch.zeros((), device=device)
        l_sp = torch.stack(step_pen).mean()  if step_pen  else torch.zeros((), device=device)
        total = l_d.to(device) + l_t.to(device) + l_sp.to(device)

        # ── L_SFT regularisation  (paper: L = L_SFT + L_disc + L_text + λ_call·N_sim)
        l_sft = torch.zeros((), device=device)
        if self.sft_loader_iter is not None and self.lambda_sft > 0:
            sft_batch  = next(self.sft_loader_iter)
            sft_inputs = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in sft_batch.items()
            }
            l_sft = model(**sft_inputs).loss * self.lambda_sft
            total = total + l_sft

        # Lightweight per-step metric logging.
        if step_rewards and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({
                "L_disc": float(l_d.detach()),
                "L_text": float(l_t.detach()),
                "L_sim":  float(l_sp.detach()),
                "L_sft":  float(l_sft.detach()),
                "avg_R":  sum(step_rewards) / len(step_rewards),
            })

        return (total, {}) if return_outputs else total


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────
    from torch.utils.data import ConcatDataset, DataLoader

    datasets = []
    if cfg.get("openworldqa_split_dir"):
        ds = OnPolicyDataset(
            split_dir   = _REPO_ROOT / cfg["openworldqa_split_dir"],
            frames_base = _REPO_ROOT / cfg["openworldqa_frames_base"],
            benchmark   = "openworldqa",
        )
        datasets.append(ds)
        print(f"[Data] OpenWorldQA on-policy: {len(ds)} samples")

    if not datasets:
        raise ValueError("No training data found. Check openworldqa_split_dir in config.")

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # ── Model ─────────────────────────────────────────────────────────────
    # Stage 1 saves only the LoRA adapter weights.  Mount them on the base
    # model with is_trainable=True so Stage-2 updates flow through the same
    # LoRA parameters.
    model_path  = _REPO_ROOT / cfg["model_name_or_path"]
    peft_config = PeftConfig.from_pretrained(str(model_path))

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype               = torch.bfloat16,
        attn_implementation = cfg.get("attn_implementation", "sdpa"),
        trust_remote_code   = True,
    )
    model = PeftModel.from_pretrained(base_model, str(model_path), is_trainable=True)
    model.gradient_checkpointing_enable()

    for name, param in model.named_parameters():
        if any(k in name for k in ("visual.", "vision_tower.", "patch_embed.")):
            param.requires_grad_(False)

    # ── World model + privileged evaluator ─────────────────────────────────
    from trajectory_gen.world_model import build_world_model
    wm_cfg: dict = {"type": cfg["world_model_type"]}
    if cfg["world_model_type"] == "helios":
        wm_cfg.update({"base_url": cfg["helios_base_url"], "model": cfg["helios_model"]})
    world_model = build_world_model(wm_cfg)

    evaluator = PrivilegedEvaluator(
        model    = cfg["evaluator_model"],
        base_url = cfg["evaluator_base_url"],
        api_key  = cfg.get("evaluator_api_key") or os.environ.get("OPENAI_API_KEY", ""),
    )

    # ── SFT regularisation loader (L_SFT term, paper Eq. 9) ────────────────
    sft_loader_iter = None
    lambda_sft      = float(cfg.get("lambda_sft", 1.0))
    if lambda_sft > 0 and cfg.get("sft_trajectory_dir"):
        from .dataset import SFTDataset, build_collate_fn as _build_collate
        sft_traj_dir = _REPO_ROOT / cfg["sft_trajectory_dir"]
        if sft_traj_dir.exists() and any(sft_traj_dir.glob("*.json")):
            sft_ds = SFTDataset(
                trajectory_dir     = sft_traj_dir,
                repo_root          = _REPO_ROOT,
                max_rollout_frames = cfg.get("max_rollout_frames", 8),
            )
            sft_loader = DataLoader(
                sft_ds,
                batch_size  = cfg.get("sft_batch_size", 2),
                shuffle     = True,
                collate_fn  = _build_collate(processor, max_length=cfg["max_length"]),
                num_workers = 2,
                pin_memory  = True,
            )
            sft_loader_iter = itertools.cycle(sft_loader)
            print(f"[Data] SFT regularisation: {len(sft_ds)} trajectories, "
                  f"lambda_sft={lambda_sft}")
        else:
            print("[Data] SFT trajectory dir empty or not found — "
                  "running Stage 2 without L_SFT regularisation.")

    # ── TrainingArguments ──────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = str(output_dir),
        max_steps                   = cfg["num_update_steps"],
        per_device_train_batch_size = cfg["batch_size"],
        gradient_accumulation_steps = cfg["gradient_accumulation_steps"],
        learning_rate               = cfg["learning_rate"],
        lr_scheduler_type           = "cosine",
        warmup_ratio                = cfg["warmup_ratio"],
        weight_decay                = 0.0,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.95,
        adam_epsilon                = 1e-8,
        max_grad_norm               = cfg["max_grad_norm"],
        bf16                        = True,
        logging_steps               = cfg["log_steps"],
        save_steps                  = cfg["save_steps"],
        save_strategy               = "steps",
        seed                        = cfg["seed"],
        report_to                   = ["tensorboard"],
        remove_unused_columns       = False,
        dataloader_num_workers      = 0,
    )

    trainer = PFOPSDTrainer(
        model            = model,
        args             = args,
        train_dataset    = dataset,
        data_collator    = lambda feats: feats,   # pass raw sample dicts through
        processing_class = processor,
        pf_cfg           = cfg,
        world_model      = world_model,
        evaluator        = evaluator,
        sft_loader_iter  = sft_loader_iter,
        lambda_sft       = lambda_sft,
    )

    trainer.train()

    # ── Final save ──────────────────────────────────────────────────────────
    final = output_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    if trainer.is_world_process_zero():
        print(f"[Done] {final}")


# ─────────────────────────────────────────────────────────────────────────────
# Logit extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_candidate_logits(
    model, processor, sample, traj, node_name, candidates, device
) -> torch.Tensor:
    """
    Return a (|C_t|,) **differentiable** score tensor for a discrete node:
    the student's log P(cand | prefix) for each candidate string.  These act
    as the logits fed to L_disc's log-softmax, so they carry gradients into
    π_θ (paper L_disc).
    """
    # Build prefix up to the decision node
    prefix = _build_prefix(traj, node_name, sample)
    log_probs = []
    for cand in candidates:
        lp = _score_string_lp(model, processor, prefix, cand, device)
        log_probs.append(lp)
    return torch.stack(log_probs)


def _score_text_node(
    model, processor, sample, traj, node_name,
    evaluator, cfg, device,
):
    """
    Sample K text candidates from the student and score each via the privileged
    evaluator.  Returns (q_values, cand_texts).

    The student log-probabilities used in L_text are NOT computed here — they
    are recomputed with gradient in the training loop via _score_string_lp so
    that L_text actually updates the policy.
    """
    K      = cfg["k_text_candidates"]
    prefix = _build_prefix(traj, node_name, sample)

    cand_texts = []
    q_values   = []

    for _ in range(K):
        text = _sample_text(
            model, processor, prefix, device,
            temperature=cfg["text_sample_temperature"],
        )
        forced_traj = dict(traj)
        if node_name == "p_sim" and forced_traj.get("simulation_attempts"):
            atts = list(forced_traj["simulation_attempts"])
            atts[-1] = dict(atts[-1], p_sim=text)
            forced_traj["simulation_attempts"] = atts
        elif node_name == "z_rel":
            forced_traj["z_rel"] = text

        reward = compute_reward(
            trajectory   = forced_traj,
            sample       = sample,
            future_paths = sample.get("future_frames", []),
            evaluator    = evaluator,
            repo_root    = _REPO_ROOT,
            lambda_sim   = cfg["lambda_sim"],
            lambda_fa    = cfg["lambda_fa"],
            lambda_fr    = cfg["lambda_fr"],
        )
        cand_texts.append(text)
        q_values.append(reward.total)

    if not cand_texts:
        return [], []

    return q_values, cand_texts


def _build_prefix(traj: dict, stop_at: str, sample: dict) -> str:
    """Build the trajectory string up to (but not including) a given node."""
    from trajectory_gen.pipeline import (
        TAG_THINK_OPEN, TAG_THINK_CLOSE,
        TAG_SIM_DEC_OPEN, TAG_SIM_DEC_CLOSE,
        TAG_SIM_PRM_OPEN, TAG_SIM_PRM_CLOSE,
        TAG_VERIFY_OPEN,  TAG_VERIFY_CLOSE,
        TAG_RELIANCE_OPEN,TAG_RELIANCE_CLOSE,
    )
    parts = []
    d = traj.get("d_sim", "no")
    if stop_at == "d_sim":
        return ""
    if traj.get("d_sim_reasoning"):
        parts.append(f"{TAG_THINK_OPEN}{traj['d_sim_reasoning']}{TAG_THINK_CLOSE}")
    parts.append(f"{TAG_SIM_DEC_OPEN}{d}{TAG_SIM_DEC_CLOSE}")

    for att in traj.get("simulation_attempts", []):
        if stop_at == "p_sim":
            break
        parts.append(f"{TAG_SIM_PRM_OPEN}{att['p_sim']}{TAG_SIM_PRM_CLOSE}")
        if att.get("z_ver_reasoning"):
            parts.append(f"{TAG_THINK_OPEN}{att['z_ver_reasoning']}{TAG_THINK_CLOSE}")
        if stop_at == "z_ver":
            break
        parts.append(f"{TAG_VERIFY_OPEN}{att['z_ver']}{TAG_VERIFY_CLOSE}")

    if stop_at == "z_rel":
        return "\n".join(parts)
    if traj.get("z_rel"):
        parts.append(f"{TAG_RELIANCE_OPEN}{traj['z_rel']}{TAG_RELIANCE_CLOSE}")
    return "\n".join(parts)


def _score_string_lp(model, processor, prefix: str, cand: str, device) -> torch.Tensor:
    """
    Compute log P(cand | prefix) under the student, **keeping gradients**.

    This is the student-view term π_θ(·|c_t, h_t^s) that L_disc and L_text
    optimise (paper Section 4.3).  It MUST be differentiable — otherwise the
    advantage-distillation objective would not update the policy at all.
    Returns a scalar tensor on the autograd graph (not a detached float).
    """
    full  = prefix + cand
    enc_f = processor.tokenizer(full,  return_tensors="pt").to(device)
    enc_p = processor.tokenizer(prefix, return_tensors="pt").to(device)
    n_ctx = enc_p["input_ids"].shape[1]

    cand_ids = enc_f["input_ids"][:, n_ctx:]
    if cand_ids.shape[1] == 0:
        # Degenerate candidate (no new tokens) — return a grad-free zero.
        return torch.zeros((), device=device)

    out = model(input_ids=enc_f["input_ids"], attention_mask=enc_f["attention_mask"])
    logprobs = torch.log_softmax(out.logits[:, n_ctx - 1: -1, :], dim=-1)
    lp = logprobs.gather(2, cand_ids.unsqueeze(-1)).squeeze(-1).sum()
    return lp


@torch.no_grad()
def _sample_text(
    model, processor, prefix: str, device, temperature: float = 1.1
) -> str:
    """Sample one text continuation from the student (no gradient needed —
    candidate *selection* is sampled, the log-prob used in L_text is recomputed
    with gradient afterwards via _score_string_lp)."""
    enc = processor.tokenizer(prefix, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens = 128,
        do_sample      = True,
        temperature    = temperature,
    )
    new_ids = out[0, enc["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 PF-OPSD self-distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",               default="train/configs/pfopsd.yaml")
    parser.add_argument("--model_name_or_path",   default=None)
    parser.add_argument("--output_dir",           default=None)
    parser.add_argument("--num_update_steps",     type=int, default=None)
    parser.add_argument("--world_model_type",     default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    for k in ("model_name_or_path", "output_dir",
              "num_update_steps", "world_model_type"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    train(cfg)


if __name__ == "__main__":
    main()

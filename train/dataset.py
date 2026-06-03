"""
dataset.py
==========
Dataset classes for PF-OPSD Stage 1 (SFT) and Stage 2 (on-policy).

Stage 1 SFT format
------------------
For each trajectory JSON produced by trajectory_gen/pipeline.py, we build
a single-turn conversation:

  • User turn   : [anchor_image] [rollout_image_1 … rollout_image_n] + text
  • Assistant   : trajectory_str  (the structured training target)

When d_sim = "yes", rollout frames from the accepted attempt are included
as additional images before the question text.  A short image-count note
is prepended to the question so the model knows which images are rollout.

Stage 2 on-policy format
------------------------
The dataset provides raw (sample, future_frames) pairs.  The training loop
generates on-policy trajectories at runtime and scores them via the
privileged evaluator.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


# ── Tag constants (must match trajectory_gen/pipeline.py) ─────────────────────
TAG_SIM_DEC_OPEN   = "<sim_decision>"
TAG_SIM_DEC_CLOSE  = "</sim_decision>"
TAG_SIM_PRM_OPEN   = "<sim_prompt>"
TAG_SIM_PRM_CLOSE  = "</sim_prompt>"
TAG_VERIFY_OPEN    = "<verify>"
TAG_VERIFY_CLOSE   = "</verify>"
TAG_RELIANCE_OPEN  = "<reliance>"
TAG_RELIANCE_CLOSE = "</reliance>"
TAG_ANSWER_OPEN    = "<answer>"
TAG_ANSWER_CLOSE   = "</answer>"

_SYSTEM_PROMPT = (
    "You are an AI assistant that reasons about physical future outcomes "
    "from visual observations.\n"
    "You may optionally use a world-model simulation to predict futures more concretely.\n\n"
    "When answering, use the following structure:\n"
    "  1. Wrap all reasoning inside <think>…</think>.\n"
    "  2. State your simulation decision: <sim_decision>yes|no</sim_decision>\n"
    "  3. If yes, output a simulation prompt: <sim_prompt>…</sim_prompt>\n"
    "  4. After each rollout, verify it: <verify>accept|reject|uncertain</verify>\n"
    "  5. State how you use the evidence: <reliance>…</reliance>\n"
    "  6. Give your final answer: <answer>A|B|C|D</answer>"
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — SFT dataset
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Loads trajectory JSONs and converts them to model-ready samples.

    Each item is a dict:
        {
            "sample_id":    str,
            "image_paths":  list[str],   # [anchor] or [anchor, rollout_1…n]
            "question_text": str,        # formatted question + options
            "target_text":  str,         # trajectory_str (assistant turn)
        }

    Actual tokenisation / processor encoding is done in the collate_fn
    so it has access to the model processor.

    Parameters
    ----------
    trajectory_dir:
        Directory containing `*.json` trajectory files (output of
        trajectory_gen/pipeline.py).
    repo_root:
        Project root for resolving relative paths stored in the JSON.
    max_rollout_frames:
        Maximum rollout frames to include per attempt (caps memory).
    d_sim_balance:
        If > 0, randomly down-sample the majority d_sim class so that the
        ratio of yes:no trajectories does not exceed this value.
        E.g. 2.0 → at most 2× as many "yes" as "no" (or vice versa).
        Set to 0 to use all samples.
    seed:
        Random seed for balancing.
    """

    def __init__(
        self,
        trajectory_dir:    str | Path,
        repo_root:         str | Path,
        max_rollout_frames: int  = 8,
        d_sim_balance:     float = 0.0,
        seed:              int   = 42,
    ):
        self.repo_root          = Path(repo_root)
        self.max_rollout_frames = max_rollout_frames

        trajectory_dir = Path(trajectory_dir)
        raw_items = sorted(trajectory_dir.glob("*.json"))
        items = [json.loads(p.read_text(encoding="utf-8")) for p in raw_items]

        if d_sim_balance > 0:
            items = _balance_d_sim(items, d_sim_balance, seed)

        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._items[idx]
        traj = item["trajectory"]

        # ── Resolve anchor frame ────────────────────────────────────────
        anchor_abs = self._abs(item["image_path"])
        image_paths = [anchor_abs]

        # ── Include rollout frames for sim path ─────────────────────────
        rollout_count = 0
        if traj["d_sim"] == "yes":
            for att in traj.get("simulation_attempts", []):
                frames = att.get("rollout_frame_paths", [])
                for f in frames[: self.max_rollout_frames]:
                    abs_f = self._abs(f)
                    if Path(abs_f).exists():
                        image_paths.append(abs_f)
                        rollout_count += 1
                # Only include rollout from the LAST attempt that has frames
                # (later attempts override earlier ones for the model input)
                if frames:
                    break

        # ── Build question text ─────────────────────────────────────────
        opts_text = "\n".join(
            f"  {k}: {v}" for k, v in item["options"].items()
        )
        question_text = f"Question: {item['question']}\n\nOptions:\n{opts_text}"
        if rollout_count > 0:
            question_text = (
                f"[Images: 1 anchor frame + {rollout_count} world-model "
                f"rollout frame(s).  The rollout frame(s) immediately follow "
                f"the anchor.]\n\n{question_text}"
            )

        return {
            "sample_id":     item["sample_id"],
            "image_paths":   image_paths,
            "question_text": question_text,
            "target_text":   item.get("trajectory_str", ""),
        }

    def _abs(self, rel: str) -> str:
        p = Path(rel)
        if p.is_absolute():
            return str(p)
        return str(self.repo_root / rel)


def _balance_d_sim(
    items: list[dict],
    max_ratio: float,
    seed: int,
) -> list[dict]:
    rng  = random.Random(seed)
    yes  = [x for x in items if x["trajectory"]["d_sim"] == "yes"]
    no   = [x for x in items if x["trajectory"]["d_sim"] != "yes"]

    if len(yes) == 0 or len(no) == 0:
        return items

    if len(yes) / len(no) > max_ratio:
        rng.shuffle(yes)
        yes = yes[: int(len(no) * max_ratio)]
    elif len(no) / len(yes) > max_ratio:
        rng.shuffle(no)
        no = no[: int(len(yes) * max_ratio)]

    combined = yes + no
    rng.shuffle(combined)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — On-policy raw dataset
# ─────────────────────────────────────────────────────────────────────────────

class OnPolicyDataset(Dataset):
    """
    Provides (sample, future_frames) pairs for Stage 2 on-policy training.

    The training loop in stage2_pfopsd.py generates trajectories on-the-fly
    from this dataset; the privileged evaluator then scores each trajectory.

    Parameters
    ----------
    split_dir:
        OpenWorldQA or VRQABench split directory containing `*.json` sample files.
    frames_base:
        Base directory for resolving frame paths (e.g. OpenWorldQA/output/frames).
    benchmark:
        "openworldqa" | "vrqabench"
    """

    def __init__(
        self,
        split_dir:   str | Path,
        frames_base: str | Path,
        benchmark:   str = "openworldqa",
    ):
        self.frames_base = Path(frames_base)
        self.benchmark   = benchmark

        split_dir = Path(split_dir)
        self._samples: list[dict] = []

        for p in sorted(split_dir.glob("*.json")):
            raw = json.loads(p.read_text(encoding="utf-8"))
            item = self._normalise(raw)
            if item:
                self._samples.append(item)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]

    def _normalise(self, raw: dict) -> dict | None:
        if self.benchmark == "openworldqa":
            video_id    = Path(raw.get("video_path", "")).name
            anchor_name = raw.get("anchor_frame", "frame_001")
            if not anchor_name.endswith(".jpg"):
                anchor_name += ".jpg"
            anchor_path = self.frames_base / video_id / anchor_name
            all_frames  = sorted(
                (self.frames_base / video_id).glob("frame_*.jpg"),
                key=lambda p: p.name,
            )
            anchor_idx   = next(
                (i for i, f in enumerate(all_frames) if f.name == anchor_name), 0
            )
            future_paths = [str(f) for f in all_frames[anchor_idx + 1:]]
            return {
                "sample_id":     raw["sample_id"],
                "benchmark":     "openworldqa",
                "image_path":    str(anchor_path),
                "future_frames": future_paths,
                "question":      raw["question"],
                "options":       raw["options"],
                "gt_answer":     raw["answer"],
                "category":      raw["category"],
            }
        # VRQABench handled similarly; extend here as needed
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (shared by both stages)
# ─────────────────────────────────────────────────────────────────────────────

def build_collate_fn(processor, max_length: int = 16384):
    """
    Returns a collate_fn that:
      1. Loads images with PIL.
      2. Applies the Qwen3.5 processor to build input_ids, attention_mask,
         pixel_values, and labels.
      3. Labels are set to -100 everywhere except the assistant turn.

    Parameters
    ----------
    processor:
        AutoProcessor loaded for the Qwen3.5 VL model.
    max_length:
        Maximum token sequence length (truncate if exceeded).
    """
    from PIL import Image

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids    = []
        all_attn_masks   = []
        all_labels       = []
        all_pixel_values = []
        all_image_grids  = []

        for item in batch:
            images = [Image.open(p).convert("RGB") for p in item["image_paths"]]

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in images]
                        + [{"type": "text", "text": item["question_text"]}]
                    ),
                },
                {"role": "assistant", "content": item["target_text"]},
            ]

            # Apply chat template — returns tokenised tensors
            encoded = processor.apply_chat_template(
                messages,
                tokenize           = True,
                add_generation_prompt = False,
                return_dict        = True,
                return_tensors     = "pt",
                padding            = False,
                truncation         = True,
                max_length         = max_length,
            )

            input_ids   = encoded["input_ids"].squeeze(0)       # (L,)
            attn_mask   = encoded["attention_mask"].squeeze(0)  # (L,)

            # Build labels: -100 everywhere except the assistant turn
            labels = _mask_non_assistant(input_ids, processor)

            # pixel_values + mm_token_type_ids (Qwen3.5 multimodal)
            all_input_ids.append(input_ids)
            all_attn_masks.append(attn_mask)
            all_labels.append(labels)

            if "pixel_values" in encoded:
                all_pixel_values.append(encoded["pixel_values"])
            if "image_grid_thw" in encoded:
                all_image_grids.append(encoded["image_grid_thw"])
            if "mm_token_type_ids" in encoded:
                if "all_mm_token_type_ids" not in dir():
                    all_mm_token_type_ids = []
                all_mm_token_type_ids.append(encoded["mm_token_type_ids"])

        # Pad sequences to the same length within the batch
        input_ids  = _pad_sequence(all_input_ids,  pad_value=processor.tokenizer.pad_token_id or 0)
        attn_masks = _pad_sequence(all_attn_masks, pad_value=0)
        labels     = _pad_sequence(all_labels,     pad_value=-100)

        out: dict[str, Any] = {
            "input_ids":      input_ids,
            "attention_mask": attn_masks,
            "labels":         labels,
        }
        if all_pixel_values:
            out["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grids:
            out["image_grid_thw"] = torch.cat(all_image_grids, dim=0)
        _mm = locals().get("all_mm_token_type_ids", [])
        if _mm:
            out["mm_token_type_ids"] = torch.cat(_mm, dim=0)
        return out

    return collate_fn


def _mask_non_assistant(
    input_ids: torch.Tensor,
    processor,
) -> torch.Tensor:
    """
    Return a labels tensor where all tokens that are NOT part of the
    assistant turn are replaced with -100.

    Detection strategy: find the last occurrence of the assistant-turn
    start token sequence and mask everything before it.
    """
    tok = processor.tokenizer
    # im_start token id (Qwen standard: <|im_start|>)
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    # "assistant" typically follows im_start as ordinary text tokens

    labels = input_ids.clone().long()

    # Walk backwards to find the last <|im_start|> … "assistant" pair
    ids = input_ids.tolist()
    last_asst_pos = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            # Check if next non-whitespace token encodes "assistant"
            # The simplest heuristic: decode the segment and check
            seg = tok.decode(ids[i : i + 5], skip_special_tokens=False)
            if "assistant" in seg.lower():
                last_asst_pos = i
                break

    if last_asst_pos >= 0:
        # Find the end of the role header (first newline after im_start + role)
        header_end = last_asst_pos
        for j in range(last_asst_pos, min(last_asst_pos + 10, len(ids))):
            seg = tok.decode([ids[j]], skip_special_tokens=False)
            if "\n" in seg:
                header_end = j + 1
                break
        labels[:header_end] = -100
    else:
        # Fallback: mask everything (should not happen with well-formed data)
        labels[:] = -100

    return labels


def _pad_sequence(
    seqs: list[torch.Tensor],
    pad_value: int,
) -> torch.Tensor:
    max_len = max(s.size(0) for s in seqs)
    padded  = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded

"""
world_model.py
==============
Unified interface for calling a generative video world model (e.g. Helios)
to produce future-rollout frames from an initial image and a text prompt.

Helios
------
Helios is the video world model used in the PF-OPSD paper.

  GitHub : https://github.com/helios-world-model/helios
           (see the paper: yuan2026helios — "Helios: A Video World Model
           for Physical Future Simulation")

  Two model variants are used in this project:
    - helios-vrbench   : fine-tuned on VR-Bench puzzle environments,
                         used for VRQABench experiments.
    - helios-general   : general-purpose physical simulation model,
                         used for OpenWorldQA experiments.

  Helios exposes an HTTP endpoint compatible with the interface expected
  by HeliosClient below.  To obtain access:
    1. Clone the Helios repository (link above).
    2. Follow the deployment instructions in its README.
    3. Set HELIOS_BASE_URL and HELIOS_API_KEY in your environment,
       or pass them via build_world_model(config).

  Request / response schema (adapt _call_api if your deployment differs):
    POST {base_url}/v1/videos/generate
    Body: {"model": "<name>", "prompt": "<text>", "image": "<base64 JPEG>"}
    Response (one of):
      {"video_url":  "<presigned URL to download MP4>"}
      {"video_b64":  "<base64-encoded MP4>"}

Implementations
---------------
  HeliosClient   — calls a live Helios HTTP endpoint.
  StubWorldModel — returns copies of the input image for offline debugging;
                   no external dependencies beyond the standard library.

Usage
-----
    from trajectory_gen.world_model import build_world_model

    # Offline debug (no Helios required):
    wm = build_world_model({"type": "stub"})
    frames = wm.generate_rollout("anchor.jpg", "simulate ball rolling off table")

    # Real Helios:
    wm = build_world_model({
        "type":     "helios",
        "base_url": "http://your-helios-endpoint",   # or set $HELIOS_BASE_URL
        "api_key":  "...",                           # or set $HELIOS_API_KEY
        "model":    "helios-vrbench",                # or "helios-general"
    })
    frames = wm.generate_rollout("anchor.jpg", "show ball falling off left edge")
"""

from __future__ import annotations

import abc
import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelClient(abc.ABC):
    """Abstract interface for generative world model clients."""

    @abc.abstractmethod
    def generate_rollout(
        self,
        initial_image_path: str,
        prompt: str,
        *,
        num_frames: int = 8,
        timeout: int = 120,
    ) -> list[str]:
        """
        Generate a video rollout from an initial image and a text prompt.

        Parameters
        ----------
        initial_image_path:
            Absolute path to the anchor / initial-condition image.
        prompt:
            Natural-language description of what to simulate.
        num_frames:
            Number of frames to extract from the generated rollout video.
        timeout:
            Maximum seconds to wait for the API call.

        Returns
        -------
        List of absolute paths to JPEG frame files, in temporal order.
        The caller is responsible for cleaning up the temporary directory
        that contains these files.

        Raises
        ------
        RuntimeError
            If the world model fails to produce a rollout.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Real Helios client
# ─────────────────────────────────────────────────────────────────────────────

class HeliosClient(WorldModelClient):
    """
    Client for the Helios video world model.

    Helios is expected to be served behind an HTTP endpoint that accepts
    a POST request with a base64-encoded initial frame and a text prompt,
    and returns an MP4 video (either as a URL or as raw base64 bytes).

    Adjust ``_call_api`` to match your deployment's exact request/response
    schema — the rest of the pipeline remains unchanged.

    Parameters
    ----------
    base_url:
        Root URL of the Helios service, e.g. ``"http://helios.internal:8080"``.
    api_key:
        Bearer token for authentication.
    model:
        Model name to pass to the API (e.g. ``"helios-vrbench"`` for the
        VR-Bench fine-tuned variant, or ``"helios-general"`` for OpenWorldQA).
    num_frames:
        Default number of frames to extract when ``num_frames`` is not
        specified at call time.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "helios-vrbench",
        num_frames: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.model    = model
        self._default_num_frames = num_frames

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def generate_rollout(
        self,
        initial_image_path: str,
        prompt: str,
        *,
        num_frames: int = 8,
        timeout: int = 120,
    ) -> list[str]:
        image_b64 = _encode_image(initial_image_path)
        mp4_bytes  = self._call_api(image_b64, prompt, timeout)

        tmpdir = Path(tempfile.mkdtemp(prefix="helios_rollout_"))
        mp4_path = tmpdir / "rollout.mp4"
        mp4_path.write_bytes(mp4_bytes)

        frames = _extract_frames_ffmpeg(str(mp4_path), str(tmpdir), num_frames)
        if not frames:
            raise RuntimeError(
                f"HeliosClient: ffmpeg extracted 0 frames from {mp4_path}"
            )
        return frames

    # ------------------------------------------------------------------
    # Internal helpers — adapt _call_api to your Helios deployment
    # ------------------------------------------------------------------

    def _call_api(self, image_b64: str, prompt: str, timeout: int) -> bytes:
        """
        POST to the Helios endpoint and return raw MP4 bytes.

        ── How to adapt this to your Helios deployment ────────────────
        See: https://github.com/helios-world-model/helios  (Helios repo)

        Helios serves a REST endpoint.  The default interface assumed here:

          POST {base_url}/v1/videos/generate
          Headers: Authorization: Bearer <api_key>
          Body (JSON):
            {
              "model":  "helios-vrbench" | "helios-general",
              "prompt": "<natural-language simulation description>",
              "image":  "<base64-encoded JPEG of the initial frame>"
            }
          Response (one of):
            {"video_url":  "<presigned URL — download the MP4 from here>"}
            {"video_b64":  "<base64-encoded MP4 bytes>"}

        If your Helios deployment uses a different schema, edit ONLY this
        method.  Everything else (frame extraction, StubWorldModel fallback,
        trajectory pipeline) stays unchanged.

        The two model variants used in the paper:
          helios-vrbench  — fine-tuned on VR-Bench maze/sokoban puzzles.
                            Use for VRQABench experiments.
          helios-general  — general physical-world simulation.
                            Use for OpenWorldQA experiments.
        ───────────────────────────────────────────────────────────────
        """
        import requests  # lazy import — not required for StubWorldModel

        response = requests.post(
            f"{self.base_url}/v1/videos/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": prompt, "image": image_b64},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        if "video_url" in data:
            dl = requests.get(data["video_url"], timeout=timeout)
            dl.raise_for_status()
            return dl.content
        elif "video_b64" in data:
            return base64.b64decode(data["video_b64"])
        else:
            raise RuntimeError(
                f"HeliosClient: unrecognised response keys {list(data.keys())}. "
                "Update HeliosClient._call_api() to match your API schema."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Stub world model (for offline debugging)
# ─────────────────────────────────────────────────────────────────────────────

class StubWorldModel(WorldModelClient):
    """
    Debug stub: returns copies of the initial image as the 'rollout'.

    Because the stub returns the same frame repeated, the privileged teacher
    will typically assign z_ver = "uncertain" or "reject" (the rollout does
    not show any future motion), which is the expected behaviour for testing
    the pipeline end-to-end without a live Helios instance.

    Parameters
    ----------
    add_noise:
        If True, adds a small amount of pixel noise to each frame to make
        consecutive frames visually distinct (requires Pillow).
        If False (default), frames are identical copies.
    """

    def __init__(self, add_noise: bool = False):
        self.add_noise = add_noise

    def generate_rollout(
        self,
        initial_image_path: str,
        prompt: str,
        *,
        num_frames: int = 8,
        timeout: int = 120,
    ) -> list[str]:
        tmpdir = Path(tempfile.mkdtemp(prefix="stub_rollout_"))
        frames: list[str] = []

        for i in range(num_frames):
            dst = str(tmpdir / f"frame_{i + 1:03d}.jpg")
            if self.add_noise:
                _copy_with_noise(initial_image_path, dst)
            else:
                shutil.copy2(initial_image_path, dst)
            frames.append(dst)

        return frames


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_world_model(config: dict) -> WorldModelClient:
    """
    Construct a WorldModelClient from a configuration dict.

    Parameters
    ----------
    config : dict
        ``type``       (str, required)  — ``"helios"`` or ``"stub"``
        ``base_url``   (str)            — required when type == "helios"
        ``api_key``    (str)            — required when type == "helios";
                                         falls back to $HELIOS_API_KEY env var
        ``model``      (str)            — Helios model name (default: "helios-vrbench")
        ``num_frames`` (int)            — default frames per rollout (default: 8)
        ``add_noise``  (bool)           — stub only: add pixel noise (default: False)

    Returns
    -------
    WorldModelClient
    """
    kind = config.get("type", "stub").lower()

    if kind == "stub":
        return StubWorldModel(add_noise=config.get("add_noise", False))

    if kind == "helios":
        api_key = config.get("api_key") or os.environ.get("HELIOS_API_KEY", "")
        if not api_key:
            raise ValueError(
                "HeliosClient requires an api_key in config or the "
                "HELIOS_API_KEY environment variable."
            )
        return HeliosClient(
            base_url   = config["base_url"],
            api_key    = api_key,
            model      = config.get("model", "helios-vrbench"),
            num_frames = config.get("num_frames", 8),
        )

    raise ValueError(
        f"Unknown world model type: {kind!r}. "
        "Supported values: 'helios', 'stub'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    """Return a base64-encoded string of the image at *path*."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _extract_frames_ffmpeg(
    mp4_path: str,
    out_dir: str,
    num_frames: int,
) -> list[str]:
    """
    Extract *num_frames* evenly-spaced JPEG frames from an MP4 using ffmpeg.

    Requires ``ffmpeg`` to be on PATH.
    """
    out_pattern = str(Path(out_dir) / "frame_%03d.jpg")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", mp4_path,
        "-vf", f"thumbnail={num_frames},setpts=N/TB",
        "-frames:v", str(num_frames),
        "-q:v", "2",
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fall back to simple fps-based extraction
        cmd_fallback = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp4_path,
            "-vf", f"fps={num_frames}/10",   # assume ~10s video
            "-q:v", "2",
            out_pattern,
        ]
        subprocess.run(cmd_fallback, capture_output=True)

    frames = sorted(Path(out_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames[:num_frames]]


def _copy_with_noise(src: str, dst: str) -> None:
    """
    Copy an image and add a small amount of random pixel noise.
    Requires Pillow; falls back to a plain copy if Pillow is not available.
    """
    try:
        import random
        from PIL import Image

        img    = Image.open(src).convert("RGB")
        pixels = img.load()
        w, h   = img.size
        n_noisy = max(1, w * h // 500)

        for _ in range(n_noisy):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, r + random.randint(-15, 15))),
                max(0, min(255, g + random.randint(-15, 15))),
                max(0, min(255, b + random.randint(-15, 15))),
            )
        img.save(dst, "JPEG", quality=92)

    except ImportError:
        shutil.copy2(src, dst)

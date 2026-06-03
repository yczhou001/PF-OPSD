"""
losses.py
=========
PF-OPSD Stage 2 loss functions.

Two loss terms (paper Section 4.3):

  L_disc  — KL divergence at DISCRETE decision nodes (d_sim, z_ver, y).
             Student minimises KL(sg[q*_t(·)] || π_θ(·|c_t, h_t^s))
             where q*_t is the advantage-reweighted privileged target.

  L_text  — Weighted log-likelihood at TEXT nodes (p_sim, z_rel).
             Student minimises -Σ_k w_{t,k} log π_θ(a_k | c_t, h_t^s)
             where weights are softmax of advantages.

Full objective (Eq. 9):
    L = L_SFT + L_disc + L_text + λ_call * E[N_sim]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Advantage computation (paper Section 4.3, Eq. 7–8)
# ─────────────────────────────────────────────────────────────────────────────

def compute_advantages(
    q_values:     Sequence[float],
    priv_probs:   Sequence[float] | None = None,
    node_type:    str = "disc",
) -> list[float]:
    """
    Compute per-action advantages A_t+(a) = Q_t+(a) - V_t+.

    For discrete nodes (node_type="disc"):
        V_t+ = Σ_a P_t+(a) * Q_t+(a)      (Eq. 7)
        where P_t+(a) is the privileged evaluator's policy.
        If priv_probs is None, treat them as uniform.

    For text nodes (node_type="text"):
        V_t+ = (1/K) * Σ_k Q_t+(a_k)      (Eq. 7)

    Parameters
    ----------
    q_values:   list of Q_t+(a) for each candidate action a.
    priv_probs: privileged evaluator's probabilities P_t+(a) (disc only).
    node_type:  "disc" or "text".

    Returns
    -------
    List of advantages A_t+(a) = Q_t+(a) - V_t+.
    """
    q = list(q_values)
    K = len(q)

    if K == 0:
        return []

    if node_type == "disc":
        p = list(priv_probs) if priv_probs is not None else [1.0 / K] * K
        # normalise in case they don't sum to 1
        p_sum = sum(p)
        if p_sum > 0:
            p = [x / p_sum for x in p]
        v = sum(p[i] * q[i] for i in range(K))
    else:  # text
        v = sum(q) / K

    return [qi - v for qi in q]


# ─────────────────────────────────────────────────────────────────────────────
# Target distribution for discrete nodes (paper Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_disc_target(
    advantages:   list[float],
    priv_probs:   list[float] | None = None,
    temperature:  float = 0.5,
) -> torch.Tensor:
    """
    Compute the advantage-reweighted target distribution q*_t for one
    discrete decision node:

        q*_t(a) ∝ π+(a | c_t+, h_t^s) * exp(A_t+(a) / τ_A)

    Parameters
    ----------
    advantages:   list of A_t+(a) for each action a ∈ C_t.
    priv_probs:   P_t+(a) from the privileged evaluator.  If None, use
                  uniform distribution.
    temperature:  τ_A (paper default 0.5).

    Returns
    -------
    Normalised probability tensor of shape (|C_t|,).
    """
    K = len(advantages)
    if K == 0:
        return torch.tensor([])

    p = priv_probs if priv_probs is not None else [1.0 / K] * K
    a = advantages

    logits = torch.tensor(
        [math.log(max(p[i], 1e-9)) + a[i] / temperature for i in range(K)],
        dtype=torch.float32,
    )
    return F.softmax(logits, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# L_disc — KL divergence at discrete nodes (paper Eq. 3-4)
# ─────────────────────────────────────────────────────────────────────────────

def l_disc(
    student_logits:  torch.Tensor,
    target_dist:     torch.Tensor,
) -> torch.Tensor:
    """
    KL(sg[q*_t] || π_θ(· | c_t, h_t^s)) at one discrete decision node.

    Parameters
    ----------
    student_logits:
        Raw logits from the student model for the |C_t| candidate actions.
        Shape: (|C_t|,).
    target_dist:
        Advantage-reweighted target distribution q*_t.  Shape: (|C_t|,).
        Should sum to 1; stop-gradient is applied internally.

    Returns
    -------
    Scalar KL divergence loss (non-negative).
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    # KL(q* || π) = Σ q*(a) * log(q*(a) / π(a))
    # = Σ q* * log q* - Σ q* * log π
    # Using F.kl_div(log_π, q*) = Σ q* * (log q* - log_π) → matches our sign
    return F.kl_div(
        student_log_probs,
        target_dist.detach(),   # stop-gradient on target
        reduction="sum",
        log_target=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# L_text — weighted log-likelihood at text nodes (paper Eq. 3, 4)
# ─────────────────────────────────────────────────────────────────────────────

def compute_text_weights(
    advantages:  list[float],
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute normalised candidate weights for a text node:

        w_{t,k} = softmax(A_t+(a_k) / τ_A)_k

    Parameters
    ----------
    advantages:  list of A_t+(a_k) for K text-node candidates.
    temperature: τ_A (paper default 0.5).

    Returns
    -------
    Weight tensor of shape (K,) that sums to 1.
    """
    if not advantages:
        return torch.tensor([])
    logits = torch.tensor(advantages, dtype=torch.float32) / temperature
    return F.softmax(logits, dim=0)


def l_text(
    candidate_log_probs: torch.Tensor,
    weights:             torch.Tensor,
) -> torch.Tensor:
    """
    Weighted negative log-likelihood over K text-node candidates:

        L_text = -Σ_k w_{t,k} * log π_θ(a_k | c_t, h_t^s)

    Parameters
    ----------
    candidate_log_probs:
        Log-probabilities of each candidate token sequence under the student.
        Shape: (K,).
    weights:
        Normalised weights from compute_text_weights().  Shape: (K,).
        Stop-gradient applied internally.

    Returns
    -------
    Scalar weighted NLL loss.
    """
    return -(weights.detach() * candidate_log_probs).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation-call penalty
# ─────────────────────────────────────────────────────────────────────────────

def l_sim_penalty(n_sim: int, lambda_call: float = 0.02) -> torch.Tensor:
    """
    λ_call * E[N_sim]: encourages selective simulation.

    Parameters
    ----------
    n_sim:        number of simulation calls in the trajectory.
    lambda_call:  coefficient (paper default 0.02).

    Returns
    -------
    Scalar penalty tensor.
    """
    return torch.tensor(lambda_call * float(n_sim), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Full PF-OPSD advantage-distillation objective (paper Eq. 9 components)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PFOPSDLossComponents:
    """Breakdown of the PF-OPSD distillation loss."""
    l_disc:      torch.Tensor
    l_text:      torch.Tensor
    l_sim_pen:   torch.Tensor
    total:       torch.Tensor


def pfopsd_loss(
    disc_nodes: list[dict],
    text_nodes: list[dict],
    n_sim:      int,
    *,
    temperature:  float = 0.5,
    lambda_call:  float = 0.02,
) -> PFOPSDLossComponents:
    """
    Compute the combined PF-OPSD advantage-distillation loss.

    Parameters
    ----------
    disc_nodes: list of dicts, each with:
        {
          "student_logits": tensor (|C_t|,),
          "advantages":     list[float],
          "priv_probs":     list[float] | None,
        }
    text_nodes: list of dicts, each with:
        {
          "candidate_log_probs": tensor (K,),
          "advantages":          list[float],
        }
    n_sim:
        Total simulation calls in this trajectory.

    Returns
    -------
    PFOPSDLossComponents with individual and summed losses.
    """
    # ── L_disc ────────────────────────────────────────────────────────────
    disc_total = torch.tensor(0.0)
    for node in disc_nodes:
        target = compute_disc_target(
            advantages  = node["advantages"],
            priv_probs  = node.get("priv_probs"),
            temperature = temperature,
        )
        disc_total = disc_total + l_disc(node["student_logits"], target)

    # ── L_text ────────────────────────────────────────────────────────────
    text_total = torch.tensor(0.0)
    for node in text_nodes:
        w = compute_text_weights(node["advantages"], temperature)
        text_total = text_total + l_text(node["candidate_log_probs"], w)

    # ── Simulation penalty ────────────────────────────────────────────────
    sim_pen = l_sim_penalty(n_sim, lambda_call)

    total = disc_total + text_total + sim_pen

    return PFOPSDLossComponents(
        l_disc    = disc_total,
        l_text    = text_total,
        l_sim_pen = sim_pen,
        total     = total,
    )

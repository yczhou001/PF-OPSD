"""
train
=====
PF-OPSD training pipeline.

stage1_sft.py   — Stage 1: Protocol SFT on privileged trajectories.
stage2_pfopsd.py — Stage 2: Privileged-Future On-Policy Self-Distillation.
dataset.py      — Dataset classes for both stages.
reward.py       — Privileged evaluator + reward computation (Stage 2).
losses.py       — L_disc (KL) and L_text (weighted likelihood) losses.
"""

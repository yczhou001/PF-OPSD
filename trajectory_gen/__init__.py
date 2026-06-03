"""
trajectory_gen
==============
Stage-1 privileged-trajectory generation for PF-OPSD.

For each training sample (from OpenWorldQA or VRQABench), a privileged teacher VLM
— which can see the ground-truth future video v* and the correct answer y* — generates
a structured concrete-reasoning trajectory:

    (d_sim, p_sim, z_ver, z_ver_reasoning, z_rel, y)

These trajectories are used as supervised fine-tuning (SFT) demonstrations for the
student MLLM in Stage 1 of PF-OPSD training.

Public API
----------
    from trajectory_gen.pipeline import TrajectoryGenerator
    from trajectory_gen.world_model import build_world_model
"""

"""Construct named policies + display labels for comparative tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_policy_eval.baselines.clip_rt_policy import ClipRTTrajectoryPolicy
from robot_policy_eval.baselines.mmfuse_policy import MMFuseTrajectoryPolicy
from robot_policy_eval.baselines.openclip_policy import OpenCLIPTrajectoryPolicy
from robot_policy_eval.baselines.openvla_policy import HFVLAtrajectoryPolicy
from robot_policy_eval.baselines.stub_policies import UnintegratedStackPolicy
from robot_policy_eval.policies.base import Policy
from robot_policy_eval.policies.dummy_physio import DummyPhysioPolicy
from robot_policy_eval.policies.dummy_vla import DummyVLAPolicy
from robot_policy_eval.policies.hybrid import HybridPolicy
from robot_policy_eval.trajectory_synthesis import load_movements_yaml

DISPLAY_NAMES_DEFAULT: dict[str, str] = {
    "mmfuse": "MMFuse",
    "clip_rt": "CLIP-RT",
    "openclip_b32": "OpenCLIP (B/32)",
    "openclip_l14": "OpenCLIP (L/14)",
    "openclip_b16": "OpenCLIP (B/16)",
    "openvla": "OpenVLA (HF)",
    "hybrid_demo": "Hybrid (VLA coarse + physio refiner, demo)",
    "dummy_physio": "Dummy physio",
    "dummy_vla": "Dummy VLA",
    "hybrid_vla_physio": "Hybrid (dummy)",
    "rt1_stub": "RT-1 / RDT (stub)",
    "rt2_stub": "RT-2 / RDT (stub)",
    "saycan_stub": "SayCan (stub)",
}


def build_policies_by_name(
    names: list[str],
    *,
    repo_root: Path | str,
    movement_yaml: Path | str,
    mmfuse_checkpoint: Path | str | None = None,
    mmfuse_mode: str = "full",
    device: str | None = None,
    seed: int = 42,
) -> tuple[list[Policy], list[tuple[str, str]]]:
    """
    Returns (policies, policy_columns) where policy_columns is (internal_name, display_name)
    for table generation.
    """
    repo_root = Path(repo_root).resolve()
    movements = load_movements_yaml(movement_yaml)
    rng = np.random.default_rng(seed)
    policies: list[Policy] = []
    columns: list[tuple[str, str]] = []

    for raw in names:
        name = raw.strip()
        pol: Policy | None = None
        if name == "mmfuse":
            if not mmfuse_checkpoint:
                raise ValueError("mmfuse requires --mmfuse-checkpoint")
            pol = MMFuseTrajectoryPolicy(
                mmfuse_checkpoint,
                repo_root=repo_root,
                mode=mmfuse_mode,
                device=device,
            )
        elif name == "clip_rt":
            pol = ClipRTTrajectoryPolicy(movements, repo_root=repo_root, device=device)
        elif name == "openclip_b32":
            pol = OpenCLIPTrajectoryPolicy(movements, repo_root=repo_root, preset="b32", device=device)
        elif name == "openclip_l14":
            pol = OpenCLIPTrajectoryPolicy(
                movements, repo_root=repo_root, preset="l14", device=device, name="openclip_l14"
            )
        elif name == "openclip_b16":
            pol = OpenCLIPTrajectoryPolicy(
                movements, repo_root=repo_root, preset="b16", device=device, name="openclip_b16"
            )
        elif name == "openvla":
            pol = HFVLAtrajectoryPolicy(movements, repo_root=repo_root, device=device)
        elif name == "hybrid_demo":
            r1 = int(rng.integers(1 << 30))
            r2 = int(rng.integers(1 << 30))
            r3 = int(rng.integers(1 << 30))
            pol = HybridPolicy(
                high_level=DummyVLAPolicy(stride=2, rng=np.random.default_rng(r1)),
                refiner=DummyPhysioPolicy(rng=np.random.default_rng(r2)),
            )
            pol.name = "hybrid_demo"
        elif name == "dummy_physio":
            pol = DummyPhysioPolicy(rng=np.random.default_rng(int(rng.integers(1 << 30))))
        elif name == "dummy_vla":
            pol = DummyVLAPolicy(stride=2, rng=np.random.default_rng(int(rng.integers(1 << 30))))
        elif name == "hybrid_vla_physio":
            r1 = int(rng.integers(1 << 30))
            r2 = int(rng.integers(1 << 30))
            r3 = int(rng.integers(1 << 30))
            pol = HybridPolicy(
                high_level=DummyVLAPolicy(stride=2, rng=np.random.default_rng(r1)),
                refiner=DummyPhysioPolicy(rng=np.random.default_rng(r2)),
            )
        elif name == "rt1_stub":
            pol = UnintegratedStackPolicy("rt1_stub", delta=(0.4, -0.3, 0.9))
        elif name == "rt2_stub":
            pol = UnintegratedStackPolicy("rt2_stub", delta=(-0.4, 0.3, 0.85))
        elif name == "saycan_stub":
            pol = UnintegratedStackPolicy("saycan_stub", delta=(0.2, 0.2, 0.5))
        else:
            raise ValueError(
                f"Unknown policy key {name!r}. See robot_policy_eval.baselines.factory or --help."
            )
        display = DISPLAY_NAMES_DEFAULT.get(name, name)
        policies.append(pol)
        columns.append((pol.name, display))

    return policies, columns


def default_sota_names() -> list[str]:
    return ["mmfuse", "clip_rt", "openclip_b32", "hybrid_demo"]

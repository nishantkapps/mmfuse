"""Built-in demo policies for the cross-model CLI (replace with your real `Policy` classes)."""

from __future__ import annotations

import numpy as np

from robot_policy_eval.policies.dummy_physio import DummyPhysioPolicy
from robot_policy_eval.policies.dummy_vla import DummyVLAPolicy
from robot_policy_eval.policies.hybrid import HybridPolicy
from robot_policy_eval.policies.base import Policy


def make_builtin_policies(names: list[str], *, seed: int = 42) -> list[Policy]:
    """
    Construct named demo policies with independent RNGs (same pattern as `pipeline.default_policies`).
    Names: dummy_physio | dummy_vla | hybrid
    """
    rng = np.random.default_rng(seed)
    by_name: dict[str, Policy] = {}

    r1 = int(rng.integers(1 << 30))
    r2 = int(rng.integers(1 << 30))
    r3 = int(rng.integers(1 << 30))
    r4 = int(rng.integers(1 << 30))

    physio = DummyPhysioPolicy(rng=np.random.default_rng(r1))
    vla = DummyVLAPolicy(stride=2, rng=np.random.default_rng(r2))
    hybrid = HybridPolicy(
        high_level=DummyVLAPolicy(stride=2, rng=np.random.default_rng(r3)),
        refiner=DummyPhysioPolicy(rng=np.random.default_rng(r4)),
    )
    by_name["dummy_physio"] = physio
    by_name["dummy_vla"] = vla
    by_name["hybrid"] = hybrid

    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"Unknown built-in policy name(s): {missing}. Choose from {sorted(by_name)}")
    return [by_name[n] for n in names]

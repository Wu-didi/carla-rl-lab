from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple


_REWARD_PROFILES = ("legacy", "research_v1")


def list_reward_profiles() -> Tuple[str, ...]:
    return _REWARD_PROFILES


def research_v1_reward(
    obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any],
    desired_speed: float,
) -> Tuple[float, Dict[str, float]]:
    """Small, explicit reward function intended for research edits."""

    del done
    desired_speed = max(float(desired_speed), 1e-6)
    speed = float(obs["ego_state"][3])
    lateral_offset = abs(float(obs["lane_info"][1]))
    longitudinal_acceleration = abs(float(obs["ego_state"][5]))

    terms = {
        "reward/speed_tracking": 10.0
        * max(0.0, 1.0 - abs(speed - desired_speed) / desired_speed),
        "reward/lane_centering": -max(lateral_offset - 0.25, 0.0),
        "reward/longitudinal_comfort": -0.5
        * max(longitudinal_acceleration - 0.5, 0.0),
        "reward/collision": -200.0 if info.get("is_collision", False) else 0.0,
        "reward/off_road": -200.0 if info.get("is_off_road", False) else 0.0,
    }
    return sum(terms.values()), terms


RewardFunction = Callable[
    [Dict[str, Any], bool, Dict[str, Any]],
    Tuple[float, Dict[str, float]],
]


def build_reward_profile(name: str, desired_speed: float) -> Optional[RewardFunction]:
    if name == "legacy":
        return None
    if name == "research_v1":
        return partial(research_v1_reward, desired_speed=desired_speed)
    raise ValueError(
        "Unknown reward profile '{}'. Available profiles: {}".format(
            name, ", ".join(_REWARD_PROFILES)
        )
    )

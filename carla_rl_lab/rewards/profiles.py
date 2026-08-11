from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple


_REWARD_PROFILES = ("nocrash_v0", "research_v1", "research_v2")


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


def research_v2_reward(
    obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any],
    desired_speed: float,
) -> Tuple[float, Dict[str, float]]:
    """Progress-aware reward that makes stationary policies suboptimal."""

    del done
    desired_speed = max(float(desired_speed), 1e-6)
    speed = max(float(obs["ego_state"][3]), 0.0)
    lateral_offset = abs(float(obs["lane_info"][1]))
    longitudinal_acceleration = abs(float(obs["ego_state"][5]))
    speed_ratio = speed / desired_speed
    idle_fraction = max(0.0, 1.0 - speed / 0.5)

    terms = {
        "reward/speed_tracking": 8.0
        * max(0.0, 1.0 - abs(speed_ratio - 1.0)),
        "reward/progress": 2.0 * min(speed_ratio, 1.0),
        "reward/idle": -0.2 * idle_fraction,
        "reward/lane_centering": -2.0 * max(lateral_offset - 0.25, 0.0),
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


def nocrash_v0_reward(
    obs: Dict[str, Any],
    done: bool,
    info: Dict[str, Any],
    desired_speed: float,
) -> Tuple[float, Dict[str, float]]:
    """Transparent NoCrash-style reward used by the pixel baseline.

    The safe desired speed may use simulator truth during training, as in the
    RLAD/RLfOLD environment. It is never part of the policy observation.
    """

    del done
    maximum_speed = max(float(desired_speed), 1e-6)
    speed = float(obs["ego_state"][3])
    safe_speed = float(info.get("safe_desired_speed", maximum_speed))
    lateral_offset = abs(float(obs["lane_info"][1]))
    heading_error = abs(float(info.get("heading_error", 0.0)))
    steer_delta = abs(float(info.get("steer_delta", 0.0)))

    terms = {
        "reward/speed": 1.0 - abs(speed - safe_speed) / maximum_speed,
        "reward/position": -lateral_offset / 2.0,
        "reward/heading": -heading_error,
        "reward/steer_change": -0.1 if steer_delta > 0.01 else 0.0,
        "reward/collision": -10.0 if info.get("is_collision", False) else 0.0,
        "reward/off_road": -10.0 if info.get("is_off_road", False) else 0.0,
        "reward/red_light": -10.0 if info.get("red_light_infraction", False) else 0.0,
    }
    return sum(terms.values()), terms


def build_reward_profile(name: str, desired_speed: float) -> Optional[RewardFunction]:
    if name == "nocrash_v0":
        return partial(nocrash_v0_reward, desired_speed=desired_speed)
    if name == "research_v1":
        return partial(research_v1_reward, desired_speed=desired_speed)
    if name == "research_v2":
        return partial(research_v2_reward, desired_speed=desired_speed)
    raise ValueError(
        "Unknown reward profile '{}'. Available profiles: {}".format(
            name, ", ".join(_REWARD_PROFILES)
        )
    )

from __future__ import annotations

from typing import Optional, Tuple

from carla_rl_lab.rewards.base import RewardComposer, WeightedRewardTerm
from carla_rl_lab.rewards.terms import (
    CollisionPenalty,
    LaneCenteringPenalty,
    LongitudinalComfortPenalty,
    OffRoadPenalty,
    SpeedTrackingReward,
)


_REWARD_PROFILES = ("legacy", "research_v1")


def list_reward_profiles() -> Tuple[str, ...]:
    return _REWARD_PROFILES


def build_reward_profile(name: str, desired_speed: float) -> Optional[RewardComposer]:
    if name == "legacy":
        return None
    if name == "research_v1":
        return RewardComposer(
            [
                WeightedRewardTerm("speed_tracking", SpeedTrackingReward(desired_speed), 10.0),
                WeightedRewardTerm("lane_centering", LaneCenteringPenalty(), 1.0),
                WeightedRewardTerm("longitudinal_comfort", LongitudinalComfortPenalty(), 0.5),
                WeightedRewardTerm("collision", CollisionPenalty(), 200.0),
                WeightedRewardTerm("off_road", OffRoadPenalty(), 200.0),
            ]
        )
    raise ValueError(
        "Unknown reward profile '{}'. Available profiles: {}".format(
            name, ", ".join(_REWARD_PROFILES)
        )
    )

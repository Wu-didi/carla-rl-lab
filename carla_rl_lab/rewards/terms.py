from __future__ import annotations

from typing import Any, Dict

from carla_rl_lab.rewards.base import RewardTerm


class SpeedTrackingReward(RewardTerm):
    name = "speed_tracking"

    def __init__(self, desired_speed: float):
        self.desired_speed = max(float(desired_speed), 1e-6)

    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        speed = float(obs["ego_state"][3])
        error = abs(speed - self.desired_speed) / self.desired_speed
        return max(0.0, 1.0 - error)


class LaneCenteringPenalty(RewardTerm):
    name = "lane_centering"

    def __init__(self, free_margin: float = 0.25):
        self.free_margin = float(free_margin)

    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        lateral_offset = float(obs["lane_info"][1])
        return -max(lateral_offset - self.free_margin, 0.0)


class LongitudinalComfortPenalty(RewardTerm):
    name = "longitudinal_comfort"

    def __init__(self, free_acceleration: float = 0.5):
        self.free_acceleration = float(free_acceleration)

    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        acceleration = abs(float(obs["ego_state"][5]))
        return -max(acceleration - self.free_acceleration, 0.0)


class CollisionPenalty(RewardTerm):
    name = "collision"

    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        return -1.0 if info.get("is_collision", False) else 0.0


class OffRoadPenalty(RewardTerm):
    name = "off_road"

    def __call__(self, obs: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        return -1.0 if info.get("is_off_road", False) else 0.0

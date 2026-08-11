"""Algorithm registry for RL-CARLA.

The registry keeps the trainer independent from concrete algorithms. New
algorithms should expose a small adapter that implements the BaseAgent API.
"""

from carla_rl_lab.algorithms.base import BaseAgent
from carla_rl_lab.algorithms.registry import (
    AlgorithmSpec,
    create_agent,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)

# Import built-in algorithms so they self-register.
from carla_rl_lab.algorithms import ddpg  # noqa: F401
from carla_rl_lab.algorithms import imitation  # noqa: F401
from carla_rl_lab.algorithms import offline  # noqa: F401
from carla_rl_lab.algorithms import on_policy  # noqa: F401
from carla_rl_lab.algorithms import sac  # noqa: F401
from carla_rl_lab.algorithms import td3  # noqa: F401

__all__ = [
    "AlgorithmSpec",
    "BaseAgent",
    "create_agent",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
]

import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Algorithm
    algorithm: str = 'sac'

    # Env
    number_of_vehicles: int = 140
    number_of_walkers: int = 0
    dt: float = 0.1
    ego_vehicle_filter: str = 'vehicle.tesla.model3'
    surrounding_vehicle_spawned_randomly: bool = True
    port: int = 2000
    town: str = 'Town05'
    max_time_episode: int = 1000
    max_waypoints: int = 12
    visualize_waypoints: bool = True
    desired_speed: float = 14.0
    max_ego_spawn_times: int = 200
    view_mode: str = 'follow'  # 'top' or 'follow'
    traffic: str = 'off'  # 'on' or 'off'
    weather: str = 'ClearNoon'
    lidar_max_range: float = 50.0
    max_nearby_vehicles: int = 5
    enable_risk_field: bool = True
    risk_field_sectors: int = 12
    risk_field_alpha: float = 0.1336
    reward_profile: str = 'legacy'

    # SAC 网络与训练
    state_dim: int = 299
    hidden_dim: int = 540
    action_dim: int = 3
    action_bound: float = 1.0
    action_mode: str = 'signed_3d'  # signed_3d (compatible) or longitudinal_2d
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 4e-4
    alpha_lr: float = 1e-4
    target_entropy: float = -3.0
    network: str = 'SAC'  # 'SAC' or 'Attention_SAC'
    exploration_noise: float = 0.1
    td3_policy_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_delay: int = 2

    # On-policy (PPO / A2C)
    rollout_steps: int = 2048
    gae_lambda: float = 0.95
    policy_lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    ppo_minibatch_size: int = 64
    total_timesteps: int = 1_000_000
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Offline RL
    dataset_path: str = ''
    offline_updates: int = 100_000
    checkpoint_interval: int = 10_000
    td3_bc_alpha: float = 2.5
    cql_alpha: float = 1.0
    cql_temperature: float = 1.0
    cql_num_random: int = 10
    offline_entropy_alpha: float = 0.2
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_max_weight: float = 100.0

    # Imitation learning
    expert_dataset_path: str = ''
    discriminator_lr: float = 3e-4
    discriminator_updates: int = 1
    imitation_updates: int = 100_000

    # Replay Buffer & 训练节拍
    buffer_size: int = 600_000
    minimal_size: int = 500
    batch_size: int = 100
    max_episodes: int = 10000
    train_every_step: bool = True
    max_walker_spawn_attempts: int = 200
    checkpoint_keep: int = 5
    checkpoint_replay_buffer: bool = False

    # Experiment logging
    logger_backend: str = 'tensorboard'  # tensorboard, wandb, both, none
    run_name: str = ''
    wandb_project: str = 'carla-rl-lab'
    wandb_entity: str = ''
    wandb_mode: str = 'offline'  # online, offline, disabled

    # Attention visualization
    log_attention_image: bool = True

    # 随机种子
    seed: int = 42

    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Optional checkpoint. Empty means train from scratch.
    use_pretrained_model: bool = False
    pretrained_model_path: str = ''

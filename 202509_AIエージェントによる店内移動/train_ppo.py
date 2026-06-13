# ==============================================
# File: train_ppo.py
# PPO training with Stable-Baselines3 using MultiInputPolicy
# ==============================================

import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch as th
import torch.nn as nn

# Import the env from same folder
from sari_sandbox import make_env, SariSandboxEnv, TaskConfig


class GridAndVecExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for Dict(grid(H,W,C), inventory(K), goal(2)).
    - CNN on grid, MLP on vectors, then concat.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        grid_space = observation_space["grid"]
        n_channels = grid_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # infer cnn output size with a dummy pass
        with th.no_grad():
            dummy = th.zeros((1, n_channels, grid_space.shape[0], grid_space.shape[1]))
            cnn_out = self.cnn(dummy).shape[1]

        vec_dim = observation_space["inventory"].shape[0] + observation_space["goal"].shape[0]
        self.mlp_vec = nn.Sequential(
            nn.Linear(vec_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(cnn_out + 64, features_dim), nn.ReLU(),
        )

    def forward(self, obs) -> th.Tensor:
        grid = obs["grid"].float() / 255.0  # (B,H,W,C)
        grid = grid.permute(0, 3, 1, 2)      # (B,C,H,W)
        inv = obs["inventory"].float()
        goal = obs["goal"].float()
        vec = th.cat([inv, goal], dim=1)
        out_grid = self.cnn(grid)
        out_vec = self.mlp_vec(vec)
        return self.linear(th.cat([out_grid, out_vec], dim=1))


def train(run_name: str = None,
          total_timesteps: int = 50_000,
          n_envs: int = 4,
          grid_size: int = 9,
          n_shelves: int = 10,
          n_skus: int = 3,
          max_steps: int = 200,
          seed: int = 42,
          save_path: str = "./models"):

    os.makedirs(save_path, exist_ok=True)
    run_name = run_name or time.strftime("sari_ppo_%Y%m%d_%H%M%S")

    def _make():
        return SariSandboxEnv(TaskConfig(grid_size=grid_size, n_shelves=n_shelves, n_skus=n_skus, max_steps=max_steps), seed=seed)

    env = make_vec_env(_make, n_envs=n_envs)

    policy_kwargs = dict(
        features_extractor_class=GridAndVecExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=1024,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(save_path, f"{run_name}.zip")
    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    # Example training run
    train(total_timesteps=100_000, n_envs=4)



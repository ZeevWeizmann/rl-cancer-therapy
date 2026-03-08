import sys
import os
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import DQN
from env.tumor_env import TumorEnv


# reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


MODEL_DIR = os.path.join("results", "models")
PLOT_DIR = os.path.join("results", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


env = TumorEnv()


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=2000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    seed=seed
)


model.learn(total_timesteps=200000)


model.save(os.path.join(MODEL_DIR, "tumor_dqn"))
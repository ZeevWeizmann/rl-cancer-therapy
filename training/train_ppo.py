import sys
import os
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
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
env.reset(seed=seed)
env = Monitor(env)


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,      
    verbose=1,
    seed=seed
)


model.learn(total_timesteps=200000)


model.save(os.path.join(MODEL_DIR, "tumor_ppo"))

print("Model saved:", os.path.join(MODEL_DIR, "tumor_ppo.zip"))
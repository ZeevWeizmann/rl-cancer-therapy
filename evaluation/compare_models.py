import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import matplotlib.pyplot as plt
from env.tumor_env import TumorEnv
from stable_baselines3 import DQN, PPO
from utils.simulation import simulate
import os
from utils.plotting import (
    plot_total_tumor,
    plot_sensitive,
    plot_resistant,
    plot_dose,
    plot_phase,
    plot_scores
)

from utils.metrics import compute_global_scores

# weight for treatment penalty in global score
LAMBDA = 0.02

# ------------------------------------------------
# load RL models
# ------------------------------------------------
MODEL_DIR = os.path.join("results", "models")

dqn = DQN.load(os.path.join(MODEL_DIR, "tumor_dqn"))
ppo = PPO.load(os.path.join(MODEL_DIR, "tumor_ppo"))


# ------------------------------------------------
# simulate all policies
# ------------------------------------------------

results = {}

results["NONE"] = simulate("none")
results["MAX"] = simulate("max")
results["THRESHOLD"] = simulate("threshold")
results["DQN"] = simulate("rl", dqn)
results["PPO"] = simulate("rl", ppo)


# # ------------------------------------------------
# # compute global score
# # ------------------------------------------------

# print("\n===== GLOBAL SCORE =====\n")

# for name, (Ts, Tr, dose) in results.items():

#     tumor = Ts + Tr

#     tumor_norm = np.mean(tumor) / 1e7
#     dose_penalty = np.mean(dose)

#     score = tumor_norm + LAMBDA * dose_penalty

#     print(name)
#     print("score:", score)
#     print()

#Plotting results
PLOT_DIR = os.path.join("results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


plot_total_tumor(results, PLOT_DIR)
plot_sensitive(results, PLOT_DIR)
plot_resistant(results, PLOT_DIR)
plot_dose(results, PLOT_DIR)
plot_phase(results, PLOT_DIR)
scores = compute_global_scores(results, LAMBDA)
plot_scores(scores, PLOT_DIR)


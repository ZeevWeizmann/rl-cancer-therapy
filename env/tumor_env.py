import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.parameters import *


class TumorEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # tumor parameters
        self.alpha_s = ALPHA_S
        self.alpha_r = ALPHA_R
        self.beta = BETA
        self.K = K

        # competition
        self.c_sr = C_SR
        self.c_rs = C_RS

        # simulation
        self.dt = DT

        # treatment doses
        self.doses = DOSES
        self.action_space = spaces.Discrete(len(self.doses))

        # scaling
        self.scale = SCALE

        # state = [tumor size, growth rate]
        self.observation_space = spaces.Box(
    low=np.array([0.0, -1.0]),
    high=np.array([1.0, 1.0]),
    dtype=np.float32
)

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.Ts = TS0
        self.Tr = TR0
        self.t = 0

        tumor = self.Ts + self.Tr

        state = np.array([
            tumor / self.scale,
            0.0
        ], dtype=np.float32)

        return state, {}

    def step(self, action):

        dose = self.doses[action]

        tumor_before = self.Ts + self.Tr

        # tumor dynamics (Lotka-Volterra competition)

        dTs = (
            self.alpha_s * self.Ts *
            (1 - (self.Ts + self.c_sr * self.Tr) / self.K)
            - self.beta * dose * self.Ts
        )

        dTr = (
            self.alpha_r * self.Tr *
            (1 - (self.Tr + self.c_rs * self.Ts) / self.K)
        )

        # Euler integration

        self.Ts += self.dt * dTs
        self.Tr += self.dt * dTr

        self.Ts = max(self.Ts, 0.0)
        self.Tr = max(self.Tr, 0.0)

        tumor_after = self.Ts + self.Tr

        self.t += 1

        # relative growth rate 

        dT = (tumor_after - tumor_before) / max(tumor_before, 1)

        # # reward

        # reward = - (tumor_after / self.scale)**2
        # reward -= 0.8 * max(dT, 0)
        # reward -= DOSE_PENALTY * dose

        TARGET = 0.5 * self.K


        reward = -((tumor_after - TARGET)/self.K)**2
        reward -= 0.5 * max(dT, 0)      # штраф если опухоль растёт
        reward -= DOSE_PENALTY * dose

        # termination

        terminated = tumor_after > self.K
        truncated = self.t >= MAX_STEPS

        # observation

        state = np.array([
            tumor_after / self.scale,
            dT
        ], dtype=np.float32)

        return state, reward, terminated, truncated, {}
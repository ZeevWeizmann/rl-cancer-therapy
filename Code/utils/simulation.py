import numpy as np
from env.tumor_env import TumorEnv
from env.parameters import MAX_STEPS


def simulate(policy, model=None):

    env = TumorEnv()

    state, _ = env.reset()

    Ts = []
    Tr = []
    dose = []

    for t in range(MAX_STEPS):

        tumor = env.Ts + env.Tr

        if policy == "none":
            action = 0

        elif policy == "max":
            action = 2

        elif policy == "threshold":

            if tumor > 5e6:
                action = 2
            else:
                action = 0

        elif policy == "rl":

            action, _ = model.predict(state, deterministic=True)

        else:
            raise ValueError

        state, reward, terminated, truncated, _ = env.step(action)

        Ts.append(env.Ts)
        Tr.append(env.Tr)
        dose.append(env.doses[action])

        if terminated or truncated:
            break

    return np.array(Ts), np.array(Tr), np.array(dose)
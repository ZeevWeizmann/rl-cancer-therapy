import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_total_tumor(results, plot_dir):

    plt.figure(figsize=(10,6))

    for name, (Ts, Tr, dose) in results.items():
        tumor = Ts + Tr
        plt.plot(tumor, label=name)

    plt.title("Total tumor size")
    plt.xlabel("time")
    plt.ylabel("tumor size")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, "total_tumor.png"))

    plt.show()
    plt.close()


def plot_sensitive(results, plot_dir):

    plt.figure(figsize=(10,6))

    for name, (Ts, Tr, dose) in results.items():
        plt.plot(Ts, label=name)

    plt.title("Sensitive tumor cells (Ts)")
    plt.xlabel("time")
    plt.ylabel("Ts")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, "sensitive_cells.png"))

    plt.show()
    plt.close()


def plot_resistant(results, plot_dir):

    plt.figure(figsize=(10,6))

    for name, (Ts, Tr, dose) in results.items():
        plt.plot(Tr, label=name)

    plt.title("Resistant tumor cells (Tr)")
    plt.xlabel("time")
    plt.ylabel("Tr")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, "resistant_cells.png"))

    plt.show()
    plt.close()


def plot_dose(results, plot_dir):

    plt.figure(figsize=(10,6))

    for name, (Ts, Tr, dose) in results.items():

        if name in ["DQN", "PPO"]:
            plt.step(range(len(dose)), dose, label=name)

    plt.title("RL treatment policies")
    plt.xlabel("time")
    plt.ylabel("dose")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, "treatment_schedule.png"))

    plt.show()
    plt.close()


def plot_phase(results, plot_dir):

    plt.figure(figsize=(8,6))

    for name, (Ts, Tr, dose) in results.items():
        plt.plot(Ts, Tr, label=name)

    plt.title("Tumor evolution: sensitive vs resistant cells")

    plt.xlabel("Sensitive cells (Ts)")
    plt.ylabel("Resistant cells (Tr)")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, "phase_plot.png"))

    plt.show()
    plt.close()


def plot_scores(scores, plot_dir):

    names = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(8,6))

    plt.bar(names, values)

    plt.title("Global treatment score")
    plt.xlabel("Policy")
    plt.ylabel("Score")

    plt.savefig(os.path.join(plot_dir, "global_score.png"))

    plt.show()
    plt.close()

def plot_learning_curves(log_files, plot_dir):

    plt.figure(figsize=(10,6))

    for name, log_file in log_files.items():

        df = pd.read_csv(log_file, skiprows=1)

        rewards = df["r"]
        lengths = df["l"]
        timesteps = lengths.cumsum()

        rolling = rewards.rolling(50).mean()

        plt.plot(timesteps, rolling, linewidth=2, label=name)

    plt.title("Learning curves")
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward")

    plt.legend()

    plt.savefig(os.path.join(plot_dir, "learning_curves.png"))

    plt.show()
    plt.close()
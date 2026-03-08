import numpy as np


def compute_global_scores(results, LAMBDA):

    scores = {}

    for name, (Ts, Tr, dose) in results.items():

        tumor = Ts + Tr

        final_tumor = tumor[-1] / 1e7
        dose_penalty = np.mean(dose)

        score = final_tumor + LAMBDA * dose_penalty

        scores[name] = score

    return scores
from betalearn import *
import numpy as np
import matplotlib.pyplot as plt

# I know I know. But this is a quick hack

TRUE_THETA = 0.3
LENGTH = 32
SAMPLES = 8
ARR_SIZE = 32
NUM_THETAS = 32


def get_improvement_ts(true_theta=0.3, length=16, samples=8):
    ls = LearningSequence(
        BetaPrior(4, randoms=[50, 20], stubborns=False),
        EvidenceStream(true_theta, length, samples),
        totev_alpha=0.01,
        permuted_evidence=False,
    )
    # This will be an array of values
    return ls.totev_alpha_spread_ts, ls.totev_alpha_dist_cover


# TODO: get both improvement graphs out (i.e. plus GC)


def get_array_imp_ts(
    array_size=16,
    true_theta=TRUE_THETA,
):
    spread_list, cover_list = [], []
    for i in range(array_size):
        spread, cover = get_improvement_ts(true_theta=true_theta, length=LENGTH)
        spread_list.append(spread)
        cover_list.append(cover)
    return spread_list, cover_list


def across_thetas(array_size=ARR_SIZE, num_thetas=NUM_THETAS):
    spread_list, cover_list = [], []
    for i in np.linspace(0, 1, num_thetas):
        spread, cover = get_array_imp_ts(true_theta=i)
        spread_list.append(spread)
        cover_list.append(cover)
        # flatten list:
        flat_spread = [item for sublist in spread_list for item in sublist]
        flat_cover = [item for sublist in cover_list for item in sublist]
    return np.array(flat_spread), np.array(flat_cover)


improvements_spread, improvements_cover = across_thetas()
yvals_spread = np.mean(improvements_spread, axis=0)
percentiles_spread = np.percentile(improvements_spread, [2.5, 97.5], axis=0)
yerr_spread = np.abs(percentiles_spread - yvals_spread)
yvals_cover = np.mean(improvements_cover, axis=0)
percentiles_cover = np.percentile(improvements_cover, [2.5, 97.5], axis=0)
yerr_cover = np.abs(percentiles_cover - yvals_cover)
xvals = range(LENGTH + 1)
plt.errorbar(xvals, yvals_spread, yerr=yerr_spread, label="Spread of values")
plt.errorbar(xvals, yvals_cover, yerr=yerr_cover, label="Coverage of true value")
plt.legend()
plt.savefig("avg-improvement.png")
# plt.show()

import numpy as np


def nanmean_after_z_score(results, axis=1, threshold=3):
    mu = np.nanmean(results, axis=axis, keepdims=True)
    std = np.nanstd(results, axis=axis, keepdims=True)
    z_score = (np.abs(results - mu)) / std
    mask = z_score < threshold
    masked_results = np.where(mask, results, np.nan)
    return np.nanmean(masked_results, axis=axis),np.nanstd(masked_results, axis=axis)

    # return

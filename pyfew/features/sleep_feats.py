import numpy as np
import scipy.stats as stats
import pandas as pd
import math

"""
Obtain 39 sleep feats, following https://www.nature.com/articles/s41598-020-79217-x.pdf
"""


def get_pre_mean_diff(data, index, num_ele):
    pre_eles = data[index - num_ele : index]
    pre_mean = np.mean(pre_eles)
    return pre_mean - data[index]


def get_next_mean_diff(data, index, num_ele):
    next_eles = data[index + 1 : index + num_ele + 1]
    next_mean = np.mean(next_eles)
    return data[index] - next_mean


def moving_sum(a, n=30 * 60 * 30):
    # n = 30 min x 60 sec x 30 hz
    # since n is even, so we need to shift to left or right by 1
    half_win_len = int(n / 2)
    a = np.pad(a, pad_width=[half_win_len, half_win_len])
    ret = np.cumsum(a, dtype=float)
    ret[half_win_len:-half_win_len] = ret[n:] - ret[:-n]
    return ret[half_win_len - 1 : -half_win_len - 1]


def moving_avg(a, n=30 * 60 * 30):
    # n = 30 min x 60 sec x 30 hz
    # since n is even, so we need to shift to left or right by 1
    win_sum = moving_sum(a, n)

    # will have to apply division considering boundary condiiton
    half_win_len = int(n / 2)
    win_sum[half_win_len:-half_win_len] = win_sum[half_win_len:-half_win_len] / n
    for i in range(half_win_len):
        win_sum[i] = win_sum[i] / n
        win_sum[-i - 1] = win_sum[-i - 1] / n
    return win_sum


def get_stats_measures(signal, signal_name="signal"):
    """
    Obtain seven stat measure for a sleep signal
    signal: N x 1: N = sample_rate * window length
    """
    feats = {
        signal_name + "_mean": np.mean(signal),
        signal_name + "_std": np.std(signal),
        signal_name + "_min": np.min(signal),
        signal_name + "_max": np.max(signal),
        signal_name + "_mad": stats.median_abs_deviation(signal),
        signal_name + "_entropy20": stats.entropy(np.histogram(signal, bins=20)[0]),
        signal_name + "_entropy200": stats.entropy(np.histogram(signal, bins=200)[0]),
    }
    return feats


def win2frame(data):
    # data (narray) of shape M x 3 x N: N = sample_rate * window_len
    # M is the epoch count
    # output long_format (narray) of shape MN x 3
    x = data[:, 0, :]
    y = data[:, 1, :]
    z = data[:, 2, :]
    long_format = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    return long_format


def get_enmo(x, y, z):
    x_sq = x ** 2
    y_sq = y ** 2
    z_sq = z ** 2
    tmp = np.sqrt(x_sq + y_sq + z_sq) - 1
    enmo = np.maximum(0, tmp)
    return enmo, x_sq, y_sq


def get_LISD(enmo):
    pre_activity_count = np.maximum(0, enmo - 0.02)

    win_len = 10  # min
    activity_count = moving_sum(pre_activity_count, n=win_len * 60 * 30)

    LIDS = 100.0 / (activity_count + 1)
    win_len = 30  # min
    LIDS = moving_avg(LIDS, n=win_len * 60 * 30)
    return LIDS


def get_epoch_feats(enmo, angleZ, LIDS, epoch_len=30, sample_rate=30):
    # Get stats at epoch level
    # Epoch_len (sec)
    # Sample_len (sec)
    enmo = enmo.reshape(-1, epoch_len * sample_rate)
    angleZ = angleZ.reshape(-1, epoch_len * sample_rate)
    LIDS = LIDS.reshape(-1, epoch_len * sample_rate)

    enmo_feats = pd.DataFrame([get_stats_measures(x, signal_name="enmo") for x in enmo])
    angleZ_feats = pd.DataFrame(
        [get_stats_measures(x, signal_name="angleZ") for x in angleZ]
    )
    LIDS_feats = pd.DataFrame([get_stats_measures(x, signal_name="LIDS") for x in LIDS])

    merged = pd.merge(
        left=enmo_feats,
        left_index=True,
        right=angleZ_feats,
        right_index=True,
        how="inner",
    )
    merged = pd.merge(
        left=merged, left_index=True, right=LIDS_feats, right_index=True, how="inner"
    )
    return merged


def getInterEpochFeat(signal_mean, signal_name):
    # This only works when window size is 30sec
    # default to 0 at boundary
    # signale_mean (narray)
    Prev30Diff = []
    Next30Diff = []
    Prev60Diff = []
    Next60Diff = []
    Prev120Diff = []
    Next120Diff = []

    epoch_len = 30
    nrow_30 = int(30 / epoch_len)
    nrow_60 = int(60 / epoch_len)
    nrow_120 = int(120 / epoch_len)

    for i in range(len(signal_mean)):
        if i < nrow_30:
            Prev30Diff.append(0)
        else:
            Prev30Diff.append(get_pre_mean_diff(signal_mean, i, nrow_30))

        if i < nrow_60:
            Prev60Diff.append(0)
        else:
            Prev60Diff.append(get_pre_mean_diff(signal_mean, i, nrow_60))

        if i < nrow_120:
            Prev120Diff.append(0)
        else:
            Prev120Diff.append(get_pre_mean_diff(signal_mean, i, nrow_120))

        if i + nrow_30 >= len(signal_mean):
            Next30Diff.append(0)
        else:
            Next30Diff.append(get_next_mean_diff(signal_mean, i, nrow_30))

        if i + nrow_60 >= len(signal_mean):
            Next60Diff.append(0)
        else:
            Next60Diff.append(get_next_mean_diff(signal_mean, i, nrow_60))

        if i + nrow_120 >= len(signal_mean):
            Next120Diff.append(0)
        else:
            Next120Diff.append(get_next_mean_diff(signal_mean, i, nrow_120))

    tmp_feats = {
        signal_name + "Prev30diff": Prev30Diff,
        signal_name + "Prev60diff": Prev60Diff,
        signal_name + "Prev120diff": Prev120Diff,
        signal_name + "Next30diff": Next30Diff,
        signal_name + "Next60diff": Next60Diff,
        signal_name + "Next120diff": Next120Diff,
    }
    tmp_df = pd.DataFrame(tmp_feats)
    return tmp_df


def sleep_features(xyz, sample_rate, win_size=30):
    # 0. transform everything into MN x 3
    xyz = win2frame(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 1. enmo
    enmo, x_sq, y_sq = get_enmo(x, y, z)

    # 2. angle z
    angleZ = np.arctan(z / (np.sqrt(x_sq + y_sq))) * 180 / math.pi

    # 3. Locomotor Inactivity during sleep (LIDS)
    LIDS = get_LISD(enmo)

    merged = get_epoch_feats(
        enmo, angleZ, LIDS, sample_rate=sample_rate, epoch_len=win_size
    )

    return merged


def channel_last2first(my_data):
    n = len(my_data)

    x = my_data[:, :, 0]
    y = my_data[:, :, 1]
    z = my_data[:, :, 2]

    x = x.reshape(n, 1, -1)
    y = y.reshape(n, 1, -1)
    z = z.reshape(n, 1, -1)

    data = np.concatenate((x, y, z), axis=1)
    return data


def get_all_sleep_feats(data, sample_rate=30):
    """
    data (narray) of shape M x 3 x N: N = sample_rate * window_len (sec)
    M is the total number of windows we hvae
    output: M x 36

    https://www.nature.com/articles/s41598-020-79217-x.pdf

    Three signal features
    Arm z
    ENMO
    LIDS
    36 Features in total 12 for each
    """
    if data.shape[1] != 3:
        data = channel_last2first(data)
    epoch_feats = sleep_features(data, sample_rate)

    enmo_mean = epoch_feats["enmo_mean"].to_numpy()
    anglez_mean = epoch_feats["angleZ_mean"].to_numpy()
    LIDS_mean = epoch_feats["LIDS_mean"].to_numpy()

    enmo_df = getInterEpochFeat(enmo_mean, "enmo")
    anglez_df = getInterEpochFeat(anglez_mean, "angleZ")
    LIDS_df = getInterEpochFeat(LIDS_mean, "LIDS")

    epoch_feats = pd.merge(
        left=epoch_feats, left_index=True, right=enmo_df, right_index=True, how="inner"
    )
    epoch_feats = pd.merge(
        left=epoch_feats,
        left_index=True,
        right=anglez_df,
        right_index=True,
        how="inner",
    )

    epoch_feats = pd.merge(
        left=epoch_feats, left_index=True, right=LIDS_df, right_index=True, how="inner"
    )

    return epoch_feats

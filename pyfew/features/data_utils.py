import numpy as np
import pandas as pd
from datetime import datetime
import time
from pyfew.features.core import extract_features


def get_ideal_times(times, ideal_times_stamps):
    time_dict = dict(zip(times, np.arange(len(times))))
    ideal_times_stamps = np.array(
        list(filter(lambda my_time: my_time in time_dict, ideal_times_stamps))
    )

    time_idx = [time_dict[my_time] for my_time in ideal_times_stamps]
    time_idx.append(len(times))  # for the last window
    time_idx = np.array(time_idx)
    return ideal_times_stamps, time_idx


def transform_data_with_time_check(acc, times, sample_rate, window_length):
    # 1. Optional because this step takes a long to verify the time continuity
    start_time = times[0]
    current_start_time = start_time
    end_time = times[-1]

    ideal_times_stamps = []
    # 1. Find all the existing ideal timestamps
    while current_start_time <= end_time:
        ideal_times_stamps.append(current_start_time)
        current_start_time += np.timedelta64(window_length, "s")

    # 2. Find out which ideal timestamps exist
    ideal_times_stamps, time_idx = get_ideal_times(times, ideal_times_stamps)

    idx_diff = np.diff(time_idx)
    time_idx = time_idx[:-1]
    time_filter = np.array(idx_diff == (sample_rate * window_length))
    ideal_times_stamps = ideal_times_stamps[time_filter]
    time2keep_idx = time_idx[time_filter]
    final_data = [
        acc[idx : idx + sample_rate * window_length, :] for idx in time2keep_idx
    ]

    final_data = np.array(final_data)
    final_times = np.array(ideal_times_stamps)
    print("Transformed the data into shape: %s" % str(final_data.shape))
    return final_data, final_times


def process_overlap(
    data, my_times, window_overlap, acc, times, sample_rate, window_length
):
    if window_overlap > 0:
        start_delay = sample_rate * window_overlap
        acc = acc[start_delay:, :]
        times = times[start_delay:]
        data_overlap, my_times_overlap = transform_data_with_time_check(
            acc, times, sample_rate, window_length
        )
        data = np.concatenate((data, data_overlap))
        my_times = np.concatenate((my_times, my_times_overlap))
    return data, my_times


def load_data(
    data,
    window_length,
    sample_rate=30,
    window_overlap=0,
    time_column="time",
    time_format="%Y-%m-%d %H:%M:%S.%f",
):
    """Load and convert the tri-axial data in epoch data

    Load the wearable data in a format that is suitable for feature extraction
    Given a consecutive sequence of tri-axial data, we will need to bin the readings into smaller chunks
    of WINDOW_LENGTH with an overlap of WINDOW_OVERLAP. If the time column is not provided, we will assume that
    there is no discontinuity in the sequence provided.

    A toy example can be found below:
    time,x,y,z,T
    2014-05-07 13:29:51.000,0.35189518,-0.44426882,0.92247915,20.0
    2014-05-07 13:29:51.020,0.28776413,-0.40227732,0.8134708,20.0
    2014-05-07 13:29:51.040,0.33098006,-0.3132904,0.8620407,20.0

    :param str/pandas.df data:
        The path to the data file or the DataFrame directly. We expect pandas.df with 3/4 columns.
    :param int window_length:
        How long each epoch should be in seconds.
    :param int window_overlap:
        What is the overlap between each epoch. This should be non-negative and smaller than window_length.
    :param int sample_rate:
        As the name suggests.
    :param str time_column:
        Column name for time.
    :param str time_format:
        Format string to parse the time column. Default value "%Y-%m-%d %H:%M:%S".
    :return:
        A suitable format dataframe for feature extraction.
    :rtype:
        pandas.DataFrame
    """

    if window_overlap < 0 or window_length <= window_overlap:
        raise Exception(
            "window_overlap should be non-negative and smaller than window_length. "
            + "Got window_length {}, window_overlap {}".format(
                window_length, window_overlap
            )
        )

    fn_start_time = time.time()
    is_np = False
    # load data if data path is provided
    if isinstance(data, str):
        print("Loading")
        if data.endswith(".npy"):
            data = np.load(data, allow_pickle=True)
            is_np = True
        elif data.endswith(".pkl"):
            data = pd.read_pickle(data)
        elif data.endswith(".csv"):
            data = pd.read_csv(data, parse_dates=[time_column])
        else:
            raise Exception(
                "Non-supported file format. Ensure the data is either .pkl or .csv"
            )
    fn_end_time = time.time()
    print("Loading completed. Took %.2f sec" % (fn_end_time - fn_start_time))
    print(len(data))

    fn_start_time = time.time()
    if is_np:
        # np
        acc = data[:, [1, 2, 3]]
        time_strings = data[:, 0]
        times = np.array(
            [datetime.strptime(date_str, time_format) for date_str in time_strings]
        )
    else:
        # pd.DataFrame
        print(data.head())
        acc = data[["x", "y", "z"]]
        acc = acc.to_numpy()
        times = data["time"].to_numpy()

    data, my_times = transform_data_with_time_check(
        acc, times, sample_rate, window_length
    )
    # currently only supports one overlap
    data, my_times = process_overlap(
        data, my_times, window_overlap, acc, times, sample_rate, window_length
    )
    print("Final shape " + str(data.shape))
    fn_end_time = time.time()
    print("Transformation took %.2f sec" % (fn_end_time - fn_start_time))
    return data, my_times


def sample_featureI(xyz, feats, feats_name="cf1"):
    feats[feats_name] = np.max(xyz, axis=0)
    return feats


def sample_featureII(xyz, feats, feats_name="cf2"):
    feats[feats_name] = np.min(xyz, axis=0)
    return feats


def sample_featureIII(xyz, feats, feats_name="cf3"):
    feats[feats_name] = np.median(xyz, axis=0)
    return feats


def main():
    data_path = "/Users/hangy/Dphil/code/pyfew/data/mini_data.csv"
    sample_rate = 50
    window_length = 30
    window_overlap = 15
    data, my_times = load_data(
        data_path, window_length, sample_rate=sample_rate, window_overlap=window_overlap
    )

    custom_features = [sample_featureI, sample_featureII, sample_featureIII]

    feats = [
        extract_features(
            epoch, sample_rate=sample_rate, custom_features=custom_features
        )
        for epoch in data
    ]
    feats = pd.DataFrame(feats)
    print(feats.columns)
    print(feats.head())


if __name__ == "__main__":
    main()

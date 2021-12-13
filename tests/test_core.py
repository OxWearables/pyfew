import pytest
import pyfew.features.core as core
import numpy as np
import os
from pyfew.features.data_utils import load_data, channel_last2first
from pyfew.features.core import extract_features


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)


def get_dummy_epoch(n=1000):
    data = np.random.rand(n, 3)
    return data


def test_invalid_sample_rate():
    my_data = get_dummy_epoch()
    sample_rate = -1
    with pytest.raises(Exception):
        core.extract_features(my_data, sample_rate=sample_rate)


def test_invalid_input_dimension():
    my_data = np.random.rand(4, 5)
    sample_rate = -1
    with pytest.raises(Exception):
        core.extract_features(my_data, sample_rate=sample_rate)


def test_invalid_feature_set_name():
    my_data = get_dummy_epoch()
    sample_rate = 10
    with pytest.raises(Exception):
        core.extract_features(my_data, sample_rate=sample_rate, feature_set="ok")


def sample_feature_i(xyz, feats, feats_name="cf1"):
    feats[feats_name] = np.max(xyz, axis=0)
    return feats


def sample_feature_ii(xyz, feats, feats_name="cf2"):
    feats[feats_name] = np.min(xyz, axis=0)
    return feats


def test_custom_features():
    sample_rate = 1500

    my_data = get_dummy_epoch(n=sample_rate)
    custom_features = [sample_feature_i, sample_feature_ii]
    feats = core.extract_epoch_features(
        my_data,
        sample_rate=sample_rate,
        custom_features=custom_features,
        feature_set="minimal",
    )
    assert "cf1" in feats
    assert "cf2" in feats


def test_features():
    sample_rate = 1500

    my_data = get_dummy_epoch(n=sample_rate)
    feats = core.extract_epoch_features(
        my_data, sample_rate=sample_rate, feature_set="minimal"
    )
    assert len(feats) == 14


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_feature_all(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = 50
    window_length = 30
    window_overlap = 15

    data, my_times = load_data(
        data_path, window_length, sample_rate=sample_rate, window_overlap=window_overlap
    )
    feats = extract_features(data, feature_set="full", sample_rate=sample_rate)

    assert len(feats.columns) == 65


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_feature_all_channel_first(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = 50
    window_length = 30
    window_overlap = 15

    data, my_times = load_data(
        data_path, window_length, sample_rate=sample_rate, window_overlap=window_overlap
    )
    print(data.shape)
    data = channel_last2first(data, sample_rate=sample_rate, epoch_len=window_length)
    feats = extract_features(
        data, feature_set="full", sample_rate=sample_rate, is_channel_last=False
    )

    assert len(feats.columns) == 65


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_sleep_feature_fail(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = 50
    window_length = 35
    window_overlap = 15

    data, my_times = load_data(
        data_path, window_length, sample_rate=sample_rate, window_overlap=window_overlap
    )
    with pytest.raises(Exception):
        _ = extract_features(data, feature_set="full", sample_rate=sample_rate)

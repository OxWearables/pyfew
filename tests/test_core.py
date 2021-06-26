import pytest
import pyfew.features.core as core
import numpy as np


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
        core.extract_features(my_data, sample_rate=sample_rate, feature_set='ok')


def sample_feature_i(xyz, feats, feats_name='cf1'):
    feats[feats_name] = np.max(xyz, axis=0)
    return feats


def sample_feature_ii(xyz, feats, feats_name='cf2'):
    feats[feats_name] = np.min(xyz, axis=0)
    return feats


def test_custom_features():
    sample_rate = 1500

    my_data = get_dummy_epoch(n=sample_rate)
    custom_features = [sample_feature_i, sample_feature_ii]
    feats = core.extract_features(my_data, sample_rate=sample_rate, custom_features=custom_features)
    assert 'cf1' in feats
    assert 'cf2' in feats


def test_features():
    sample_rate = 1500

    my_data = get_dummy_epoch(n=sample_rate)
    feats = core.extract_features(my_data, sample_rate=sample_rate)
    np.save('data/feats.npy', feats)
    assert len(feats) == 27


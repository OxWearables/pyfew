import pytest
import pyfew.features.data_utils as data_utils


def test_invalid_sample_rate():
    data_path = 'data/mini_data.csv'
    sample_rate = -1
    window_length = 30
    window_overlap = 45
    with pytest.raises(Exception):
        _, _ = data_utils.load_data(data_path, window_length, sample_rate=sample_rate,
                                    window_overlap=window_overlap)


def test_big_overlap():
    data_path = 'data/mini_data.csv'
    sample_rate = 50
    window_length = 30
    window_overlap = 45
    with pytest.raises(Exception):
        _, _ = data_utils.load_data(data_path, window_length, sample_rate=sample_rate,
                                    window_overlap=window_overlap)


def test_loader():
    data_path = 'data/mini_data.csv'
    sample_rate = 50
    window_length = 30
    window_overlap = 15
    data, my_times = data_utils.load_data(data_path, window_length, sample_rate=sample_rate,
                                          window_overlap=window_overlap)
    assert len(data) == 132
    assert data.shape[1] == 1500
    assert data.shape[2] == 3

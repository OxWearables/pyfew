import pytest
import pyfew.features.data_utils as data_utils
import os

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_invalid_sample_rate(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = -1
    window_length = 30
    window_overlap = 45
    with pytest.raises(Exception):
        _, _ = data_utils.load_data(
            data_path,
            window_length,
            sample_rate=sample_rate,
            window_overlap=window_overlap,
        )


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_big_overlap(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = 50
    window_length = 30
    window_overlap = 45
    with pytest.raises(Exception):
        _, _ = data_utils.load_data(
            data_path,
            window_length,
            sample_rate=sample_rate,
            window_overlap=window_overlap,
        )


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "mini_data.csv"))
def test_loader(datafiles):
    data_path = str(datafiles.listdir()[0])
    sample_rate = 50
    window_length = 30
    window_overlap = 15
    data, my_times = data_utils.load_data(
        data_path, window_length, sample_rate=sample_rate, window_overlap=window_overlap
    )
    assert len(data) == 132
    assert data.shape[1] == 1500
    assert data.shape[2] == 3

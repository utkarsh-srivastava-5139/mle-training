import os
import tarfile

import pytest
import requests


@pytest.fixture(scope="session")
def data_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="session")
def download_dataset(data_dir):
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    filename = os.path.join(data_dir, "data.tar.gz")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(data_dir)


def test_dataset_exists(data_dir, download_dataset):
    assert os.path.isfile(os.path.join(data_dir, "housing.csv"))


def test_dataset_format(data_dir, download_dataset):
    with open(os.path.join(data_dir, "housing.csv")) as f:
        header = f.readline().strip()
        assert (
            header
            == "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity"
        )

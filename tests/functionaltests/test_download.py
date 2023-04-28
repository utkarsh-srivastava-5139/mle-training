# Import necessary libraries
import os
import tarfile

import pytest
import requests


# Define a session-level fixture to create a temporary directory to store data
@pytest.fixture(scope="session")
def data_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


# Define a session-level fixture to download the dataset from a URL
# and extract it in the data directory
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


# Define a test to check if the dataset file exists in the data directory
def test_dataset_exists(data_dir, download_dataset):
    assert os.path.isfile(os.path.join(data_dir, "housing.csv"))


# Define a test to check the format of the dataset file in the data directory
def test_dataset_format(data_dir, download_dataset):
    with open(os.path.join(data_dir, "housing.csv")) as f:
        header = f.readline().strip()
        assert (
            header
            == "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity"
        )

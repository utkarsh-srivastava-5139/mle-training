# Test script that tests the income_cat_proportions() function
# from the housinglib library

import os.path as op
import sys

import pandas as pd
import pytest

from housinglib import income_cat_proportions as icp

HERE = op.dirname(op.abspath(__file__))
test_path = op.join(HERE, "..", "..")
data_path = op.join(HERE, "..", "testdata")
sys.path.append(test_path)


value = pd.read_csv(data_path + "/test_data.csv")


# Define a parameterized test
@pytest.mark.parametrize("test_input,expected", [(value, [0.4, 0.4, 0.2])])
def test_income_cat_proportions(test_input, expected):
    assert list(icp(test_input)) == expected

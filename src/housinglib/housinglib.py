import logging
import os
import os.path as op
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def fetch_housing_data(housing_url, housing_path):
    """fetch_housing_data

    This function fetches the data from the input url and downloads the file in the provided local path.

    Parameters
    ----------
            housing_url:
                    link to download the dataset
            housing_path:
                    Path to save the datasets
    """
    logger = logging.getLogger(__name__)
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info("Fetched the data from the source")


def load_housing_data(housing_path):
    """load_housing_data

    This function loads the data from the in put path and returns the pandas dataframe.

    Parameters
    ----------
            housing_path:
                    Path to save the datasets

    Returns
    -------
    data_frame from the input file.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def load_data(housing_path):
    """load_data

    load_data function loads the data to the data/raw folder in the main folder structure.

    Parameters
    ----------
            housing_path:
                    Path to save the datasets
    """
    logger = logging.getLogger(__name__)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    # HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = housing_path
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    logger.info("Loaded housing data to respective folders")


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# Create a customer transformer to add extra attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def transformation_pipeline(data_num, data):
    """transformation_pipeline

    Build a pipeline for preprocessing the data:
            - Imputing the missing cells with median
            - Using customer transformer for adding extra attributes
            - Scaling the data
            - Creating dummy variables for categorical featues using one-hot encoding

    Parameters
    ----------
            data_num:
                    Numerical features
            data:
                    Whole data containing both numerical and categorical features

    Returns
    -------
    housing_prepared
    """
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(data_num)
    cat_attribs = ["ocean_proximity"]

    # Build the full pipeline
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(data)

    return housing_prepared


def data_prep(housing_path, project_path):
    """data_prep

    data_prep function prepares the data for training and testing.

    Parameters
    ----------
            housing_path:
                    Path to save the datasets
            project_path:
                    Path to root directory of the project
    """
    logger = logging.getLogger(__name__)
    HOUSING_PATH = housing_path
    housing = load_housing_data(HOUSING_PATH)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    corr_matrix = housing.loc[:, housing.columns != "ocean_proximity"].corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    housing_prepared = transformation_pipeline(housing_num, housing)

    logger.info("Prepared training and testing set for the models")

    processed_data_path = op.join(project_path, "..", "data", "processed")

    train_set = pd.DataFrame(housing_prepared)
    housing_labels2 = pd.DataFrame(housing_labels).reset_index(drop=True)
    train_set = pd.concat([train_set, housing_labels2], axis=1)

    train_set.to_csv(os.path.join(processed_data_path, "train_set.csv"), index=False)

    strat_test_set.to_csv(
        os.path.join(processed_data_path, "test_set.csv"), index=False
    )


def load_train_data(project_path):
    """load_train_data

    Load the training data and split them into dependent and independent variables

    Parameters
    ----------
            project_path:
                    Path to root directory of the project

    Returns
    -------
    housing_prepared, housing_labels.
    """
    processed_data_path = op.join(project_path, "..", "data", "processed")

    train_set = pd.read_csv(os.path.join(processed_data_path, "train_set.csv"))
    housing_prepared = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"]

    return housing_prepared, housing_labels


def lin_reg(housing_prepared, housing_labels):
    """lin_reg

    Trains the linear regression model

    Parameters
    ----------
            housing_prepared:
                    prepared dataset for training.
            housing_labels:
                    Labels of the data frame.

    Returns
    -------
    RMSE, MAE.
    """
    logger = logging.getLogger(__name__)
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)

    logger.info("Trained linear regression model")

    return {"rmse": round(lin_rmse, 2), "mae": round(lin_mae, 2)}


def desc_tree(housing_prepared, housing_labels):
    """desc_tree

    This function trains the data on the decision tree regressor.

    Parameters
    ----------
            housing_prepared:
                    prepared dataset for training.
            housing_labels:
                    Labels of the data frame.

    Returns
    -------
    RMSE, MAE.
    """
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    return {"rmse": round(tree_rmse, 4), "mae": round(tree_mae, 4)}


def random_forest(housing_prepared, housing_labels):
    """random_forest

    This function trains the data on the random foreset regressor for different params.

    Parameters
    ----------
            housing_prepared:
                    prepared dataset for training.
            housing_labels:
                    Labels of the data frame.

    Returns
    -------
    final_model.
    """
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    return final_model


def load_test_data(project_path):
    """load_test_data

    Load the testing data

    Parameters
    ----------
            project_path:
                    Path to root directory of the project

    Returns
    -------
    test_set
    """
    processed_data_path = op.join(project_path, "..", "data", "processed")

    test_set = pd.read_csv(os.path.join(processed_data_path, "test_set.csv"))

    return test_set


def model_score(model, test_data):
    """model_score

    Load the model and runs it on the test data and calculates model's performance

    Parameters
    ----------
            model:
                    Model trained on the training data
            test_data:
                    Test data to calcualtes model's performance

    Returns
    -------
    final_mse, final_rmse
    """
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)

    X_test_prepared = transformation_pipeline(X_test_num, test_data)

    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    return final_mse, final_rmse

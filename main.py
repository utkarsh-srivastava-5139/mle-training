import logging
import os
import os.path as op
import tarfile

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
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
from sklearn.tree import DecisionTreeRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
project_path = "/home/utkarsh/TAMLEP/mle-training/scripts"


def fetch_housing_data(housing_url, housing_path):
    logger = logging.getLogger(__name__)
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logger.info("Fetched the data from the source")


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def load_data(housing_path):
    logger = logging.getLogger(__name__)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    # HERE = op.dirname(op.abspath(__file__))
    HOUSING_PATH = housing_path
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    logger.info("Loaded housing data to respective folders")


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def data_prep(housing_path, project_path):
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
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    logger.info("Prepared training and testing set for the models")

    artifact_path = op.join(project_path, "..", "artifacts")

    # save the imputer
    from pickle import dump

    dump(imputer, open(f"{artifact_path}/imputer.pkl", "wb"))

    processed_data_path = op.join(project_path, "..", "data", "processed")

    train_set = housing_prepared.copy()
    train_set[housing_labels.name] = housing_labels.values

    train_set.to_csv(os.path.join(processed_data_path, "train_set.csv"), index=False)
    strat_test_set.to_csv(
        os.path.join(processed_data_path, "test_set.csv"), index=False
    )

    return housing_prepared, housing_labels, strat_test_set, imputer


def load_train_data(project_path):
    processed_data_path = op.join(project_path, "..", "data", "processed")

    train_set = pd.read_csv(os.path.join(processed_data_path, "train_set.csv"))
    housing_prepared = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"]

    return housing_prepared, housing_labels


def lin_reg(housing_prepared, housing_labels):
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
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    return {"rmse": round(tree_rmse, 4), "mae": round(tree_mae, 4)}


def random_forest(housing_prepared, housing_labels):
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
    processed_data_path = op.join(project_path, "..", "data", "processed")
    artifact_path = op.join(project_path, "..", "artifacts")

    from pickle import load

    imputer = load(open(f"{artifact_path}/imputer.pkl", "rb"))

    test_set = pd.read_csv(os.path.join(processed_data_path, "test_set.csv"))

    return test_set, imputer


def model_score(model, test_data, imputer):
    X_test = test_data.drop("median_house_value", axis=1)
    y_test = test_data["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    return final_mse, final_rmse


if __name__ == "__main__":
    experiment = mlflow.set_experiment("Housing_price_prediction")
    np.random.seed(40)
    load_data(HOUSING_PATH)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # set parent run_id for all child runs
        parent_run_id = run.info.run_id
        test_size = 0.2
        strategy = "median"

        with mlflow.start_run(
            run_id=None,
            experiment_id=run.info.experiment_id,
            nested=True,
            run_name="data_prep",
        ) as child_run:
            # log any parameters or metrics relevant to data_prep
            (
                housing_prepared,
                housing_labels,
                strat_test_set,
                imputer,
            ) = data_prep(HOUSING_PATH, project_path)
            mlflow.log_param("split_ratio", test_size)
            mlflow.log_param("strategy", strategy)
            mlflow.log_metric("Number_of_rows", len(housing_prepared))

        with mlflow.start_run(
            run_id=None,
            experiment_id=run.info.experiment_id,
            nested=True,
            run_name="train_model",
        ) as child_run:
            n_estimators = [3, 5]
            max_features = [2, 4, 6]

            # run model training script
            final_model = random_forest(
                housing_prepared,
                housing_labels,
            )
            # log any parameters or metrics relevant to train_model
            mlflow.log_param("estimators", n_estimators)
            mlflow.log_param("max_features", max_features)

        with mlflow.start_run(
            run_id=None,
            experiment_id=run.info.experiment_id,
            nested=True,
            run_name="model_score",
        ) as child_run:
            # run model scoring script
            final_mse, final_rmse = model_score(final_model, strat_test_set, imputer)

            # log any parameters or metrics relevant to score_model
            mlflow.log_metric("score_model_mae", final_mse)
            mlflow.log_metric("score_model_rmse", final_rmse)

        mlflow.log_param("split_ratio", test_size)
        mlflow.log_param("strategy", strategy)
        mlflow.log_metric("Number_of_rows", len(housing_prepared))
        mlflow.log_param("estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("score_model_mae", final_mse)
        mlflow.log_metric("score_model_rmse", final_rmse)

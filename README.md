# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Development environment setup
Use the terminal or an Anaconda Prompt for the following steps:

- Create the environment from the env.yml file, by entering the following command:
    ```
    conda env create -f env.yml
    ```
    The first line of the yml file sets the new environment's name.

- Activate the new environment, by entering the following command: 
    ```
    conda activate mle-dev
    ```
- Verify that the new environment was installed correctly, by entering the following command:
    ```
    conda env list
    ```


## To install the housinglib library
First, download the wheel file from the repo, then run the following command:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps housinglib_utkarsh
```
Check if the housinglib_utkarsh package is installed by entering the following command: 
```
import housinglib_utkarsh
```

## The scripts folder contains the scripts to download data, train and check scores of the model
To load the data and create training and testing set, use the ingest_data.py script by running following command in the shell:
```
python ingest_data.py
```
To train the model using the training data, use train.py
```
python train.py
```
To see the performance of the model, use score.py
```
python score.py
```

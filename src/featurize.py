#!/usr/bin/env python3
"""Clean up inputs and add features to data set.

Author:
    Erik Johannes Husom

Date:
    2020-09-16

"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from config import DATA_FEATURIZED_PATH
from preprocess_utils import read_csv, move_column


def featurize(filepaths):
    """Clean up inputs and add features to data set.

    Args:
        filepaths (list of str): List of paths to files to process.

    """

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

    # Load parameters
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    features = params["features"]
    """Features to be engineered."""

    delete_features = params["delete"]
    """Features to be deleted before adding new features."""

    remove_features = params["remove"]
    """Features to be removed after adding new features, since they are used in
    feature engineereing.
    """

    scale = params["scale"]
    """Whether to scale the input features before feature engineering."""

    for filepath in filepaths:

        # Read csv, and delete specified columns
        df, index = read_csv(
            filepath,
            delete_columns=delete_features,
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        if scale:
            df = scale_inputs(df)

        add_features(df, features)

        # Remove columns from input. Check first if it is a list, to avoid
        # error if empty.
        if isinstance(remove_features, list):
            for col in remove_features:
                del df[col]

        # Save data
        df.to_csv(
            DATA_FEATURIZED_PATH
            / (os.path.basename(filepath).replace("restructured", "featurized"))
        )

def scale_inputs(df):
    """Scale input features.

    Args:
        df (DataFrame): Data frame containing data.

    Returns:
        scaled_df (DataFrame): Data frame containing scaled data.

    """

    # Load scaling parameters
    params = yaml.safe_load(open("params.yaml"))["scale"]
    method = params["method"]
    heartrate_min = params["heartrate_min"]
    heartrate_max = params["heartrate_max"]
    breathing_min = params["breathing_min"]
    breathing_max = params["breathing_max"]
    
    heartrate_range = heartrate_max - heartrate_min
    breathing_range = breathing_max - breathing_min

    if "heartrate" in df.columns:
        df["heartrate"] = (df["heartrate"] - heartrate_min)/heartrate_range

    if "ribcage" in df.columns:
        df["ribcage"] = (df["ribcage"] - breathing_min)/breathing_range

    if "abdomen" in df.columns:
        df["abdomen"] = (df["abdomen"] - breathing_min)/breathing_range

    return df

def add_features(df, features):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    win = 50

    if "ribcage_min" in features:
        ribcage_min = df["ribcage"].rolling(win).min()

        # Add column to data frame
        df["ribcage_min"] = ribcage_min

    if "ribcage_max" in features:
        ribcage_max = df["ribcage"].rolling(win).max()

        # Add column to data frame
        df["ribcage_max"] = ribcage_max

    if "ribcage_range" in features:

        # Calculate min, max and range
        ribcage_min = df["ribcage"].rolling(win).min()
        ribcage_max = df["ribcage"].rolling(win).max()
        ribcage_range = ribcage_max - ribcage_min

        # Add column to data frame
        df["ribcage_range"] = ribcage_range

    if "abdomen_min" in features:
        abdomen_min = df["abdomen"].rolling(win).min()

        # Add column to data frame
        df["abdomen_min"] = abdomen_min

    if "abdomen_max" in features:
        abdomen_max = df["abdomen"].rolling(win).max()

        # Add column to data frame
        df["abdomen_max"] = abdomen_max

    if "abdomen_range" in features:

        # Calculate min, max and range
        abdomen_min = df["abdomen"].rolling(win).min()
        abdomen_max = df["abdomen"].rolling(win).max()
        abdomen_range = abdomen_max - abdomen_min

        # Add column to data frame
        df["abdomen_range"] = abdomen_range


def add_feature(df, name, feature_col):
    """Adding a feature to the data set.

    Args:
        df (pandas DataFrame): Data frame to add features to.
        name (str): What to call the new feature.
        feature_col (array-like): The actual data to add to the input matrix.
        add_to_hist_matrix (bool): Whether to use the feature in as historical
            data, meaning that data points from previous time steps also will be
            included in the input matrix. If set to True, only the current data
            point will be used as input. Default=False.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    df[name] = feature_col

    print("Feature added: {}".format(name))

    return df


if __name__ == "__main__":

    featurize(sys.argv[1:])

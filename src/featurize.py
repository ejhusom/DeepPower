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
import numpy as np
import pandas as pd
import yaml

from config import DATA_FEATURIZED_PATH, DATA_PATH
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
    feature engineering.
    """

    scale = params["scale"]
    """Whether to scale the input features before feature engineering."""

    range_window = params["range_window"]
    slope_window = params["slope_window"]

    for filepath in filepaths:

        # Read csv, and delete specified columns
        df, index = read_csv(
            filepath,
            delete_columns=delete_features
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        if scale:
            df = scale_inputs(df)

        add_features(df, features, 
                range_window=range_window,
                slope_window=slope_window,
        )

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

    # Save list of features used
    pd.DataFrame(df.columns).to_csv(DATA_PATH / "input_columns.csv")

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

def add_features(df, features, 
        range_window=100,
        slope_window=10,
    ):
    """
    This function adds features to the input data, based on the arguments
    given in the features-list.

    Args:
    df (pandas DataFrame): Data frame to add features to.
    features (list): A list containing keywords specifying which features to
        add.
    range_window (int): How many time steps to use when calculating range.
    slope_window (int): How mant time steps to use when calculating slope.

    Returns:
        df (pandas DataFrame): Data frame with added features.

    """

    # Stop function of features is not a list
    if features == None:
        return 0

    if "ribcage_min" in features:
        ribcage_min = df["ribcage"].rolling(range_window).min()

        df["ribcage_min"] = ribcage_min

    if "ribcage_max" in features:
        ribcage_max = df["ribcage"].rolling(range_window).max()

        df["ribcage_max"] = ribcage_max

    if "ribcage_range" in features:

        ribcage_min = df["ribcage"].rolling(range_window).min()
        ribcage_max = df["ribcage"].rolling(range_window).max()
        ribcage_range = ribcage_max - ribcage_min

        df["ribcage_range"] = ribcage_range

    if "abdomen_min" in features:
        abdomen_min = df["abdomen"].rolling(range_window).min()

        df["abdomen_min"] = abdomen_min

    if "abdomen_max" in features:
        abdomen_max = df["abdomen"].rolling(range_window).max()

        df["abdomen_max"] = abdomen_max

    if "abdomen_range" in features:

        abdomen_min = df["abdomen"].rolling(range_window).min()
        abdomen_max = df["abdomen"].rolling(range_window).max()
        abdomen_range = abdomen_max - abdomen_min

        df["abdomen_range"] = abdomen_range

    if "ribcage_gradient" in features:

        df["ribcage_gradient"] = np.gradient(df["ribcage"])

    if "abdomen_gradient" in features:

        df["abdomen_gradient"] = np.gradient(df["abdomen"])

    if "ribcage_slope" in features:
        pass

def calculate_frequency(peaks):
    """Calculate frequency based on peaks.

    Args:
        peaks (array): Array with 1 as a peak, and 0 as no peak.

    Returns:
        freq (array): Array of frequency.

    """

    freq = []
    f = 0
    counter = 0

    for i, p in enumerate(peaks):

        if p == 1:
            f = 10 / counter
            counter = 0
        else:
            counter += 1

        freq.append(f)

    freq = np.array(freq)*60
    freq = freq.rolling(100).mean()

    return freq

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

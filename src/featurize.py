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

    params = yaml.safe_load(open("params.yaml"))["featurize"]

    features = params["features"]
    """Features to be engineered."""

    delete_features = params["delete"]
    """Features to be deleted before adding new features."""

    remove_features = params["remove"]
    """Features to be removed after adding new features, since they are used in
    feature engineereing.
    """

    for filepath in filepaths:

        # Read csv, and delete specified columns
        df, index = read_csv(
            filepath,
            delete_columns=delete_features,
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        add_features(df, features)

        # Remove columns from input.
        for col in remove_features:
            del df[col]

        df.to_csv(
            DATA_FEATURIZED_PATH
            / (os.path.basename(filepath).replace("restructured", "featurized"))
        )

        df.plot()
        plt.show()

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

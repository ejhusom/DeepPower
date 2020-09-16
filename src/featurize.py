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
    remove_features = params["remove"]

    for filepath in filepaths:

        df, index = read_csv(
            filepath,
            delete_columns=remove_features,
        )

        # Move target column to the beginning of dataframe
        df = move_column(df, column_name="power", new_idx=0)

        # self.add_features(features)

        # Save the names of the input columns
        # self.input_columns = self.df.columns
        # input_columns_df = pd.DataFrame(self.input_columns)
        # input_columns_df.to_csv(self.result_dir + self.time_id +
        #         "-input_columns.csv")

        df.to_csv(
            DATA_FEATURIZED_PATH
            / (os.path.basename(filepath).replace("restructured", "featurized"))
        )

        df.plot()
        plt.show()


if __name__ == "__main__":

    featurize(sys.argv[1:])

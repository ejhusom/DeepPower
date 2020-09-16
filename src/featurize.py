#!/usr/bin/env python3
# ============================================================================
# File:     featurize.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Add engineered features to dataset.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml

from config import Config
from preprocess_utils import *
from utils import *

def featurize(filepaths):

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    Config.DATA_FEATURIZED_PATH.mkdir(parents=True, exist_ok=True)

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

        df.to_csv(Config.DATA_FEATURIZED_PATH / (
            os.path.basename(filepath).replace("restructured", "featurized"))
        )

        df.plot()
        plt.show()


if __name__ == '__main__':

    featurize(sys.argv[1:])

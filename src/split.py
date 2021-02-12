#!/usr/bin/env python3
"""Split data into training and test set.

Author:
    Erik Johannes Husom

Date:
    2020-10-29

"""
import os
import sys

import numpy as np
import yaml

from config import DATA_SPLIT_PATH

# from config import DATA_SPLIT_TRAIN_PATH, DATA_SPLIT_TEST_PATH
from preprocess_utils import read_csv


def split(filepaths):
    """Split data into train and test set.

    Training files and test files are saved to different folders.

    Args:
        filepaths (list of str): A list of paths to files containing
            featurized data.

    """

    # Handle special case where there is only one workout file.
    if isinstance(filepaths, str) or len(filepaths) == 1:
        raise NotImplementedError("Cannot handle only one workout file.")

    DATA_SPLIT_PATH.mkdir(parents=True, exist_ok=True)
    # DATA_SPLIT_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
    # DATA_SPLIT_TEST_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["split"]

    # Parameter 'train_split' is used to find out no. of files in training set
    file_split = int(len(filepaths) * params["train_split"])

    training_files = filepaths[:file_split]
    test_files = filepaths[file_split:]

    from_test_to_training_files = []

    # Check for rest data files
    for f in test_files:
        if "-rest-" in f:
            from_test_to_training_files.append(f)

    # Move some files to test set 
    for i in range(len(from_test_to_training_files)):
        test_files.append(training_files[-1])
        training_files.remove(training_files[-1])

    # Move rest data files to training set
    for f in from_test_to_training_files:
        training_files.append(f)
        test_files.remove(f)

    # print("_______________________________________")
    # print("TRAINING")
    # for f in training_files:
    #     print(f)

    # print("TEST")
    # for f in test_files:
    #     print(f)

    for filepath in filepaths:

        df, index = read_csv(filepath)

        if filepath in training_files:
            df.to_csv(
                DATA_SPLIT_PATH
                / (os.path.basename(filepath).replace("featurized", "train"))
            )
        elif filepath in test_files:
            df.to_csv(
                DATA_SPLIT_PATH
                / (os.path.basename(filepath).replace("featurized", "test"))
            )


if __name__ == "__main__":

    np.random.seed(2020)

    split(sys.argv[1:])

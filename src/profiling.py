import os 
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import yaml

from config import DATA_RESTRUCTURED_PATH, PROFILING_PATH


def profile(filepaths):

    # Define an empty data frame.
    df = pd.DataFrame()

    # Read and append the csv files
    for filepath in filepaths:
        tmp = pd.read_csv(filepath)
        df = df.append(tmp)

    # Generate report.
    profile = ProfileReport(df, title="Profiling Analysis",
            config_file="src/profiling_config.yaml", lazy=False)

    # Create folder for profiling report
    PROFILING_PATH.mkdir(parents=True, exist_ok=True)

    # Save report to html.
    profile.to_file(PROFILING_PATH / "profile_report.html")

if __name__ == "__main__":

    profile(sys.argv[1:])

#!/usr/bin/env python3
"""Global parameters for project.

Example:

    >>> from config import *
    >>> file = DATA_PATH / "filename.txt"

Author:   Erik Johannes Husom
Created:  2020-09-16

"""

from pathlib import Path


ASSETS_PATH = Path("./assets")
DATA_PATH = ASSETS_PATH / "data"
DATA_RESTRUCTURED_PATH = DATA_PATH / "restructured"
DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
DATA_SEQUENTIALIZED_PATH = DATA_PATH / "sequentialized"
DATA_COMBINED_PATH = DATA_PATH / "combined"
DATA_SPLIT_PATH = DATA_PATH / "split"
DATA_SCALED_PATH = DATA_PATH / "scaled"
MODELS_PATH = "../models"
METRICS_PATH = "../metrics"
METRICS_FILE_PATH = "../metrics/metrics.json"

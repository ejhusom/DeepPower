#!/usr/bin/env python3
# ============================================================================
# File:     combine.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Combine sequentialized data from multiple workouts into one data set.
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml

from config import Config
from preprocess_utils import *
from utils import *


def combine(filepaths):

    # If filepaths is a string (e.g. only one filepath), wrap this in a list
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    Config.DATA_COMBINE_PATH.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(open("params.yaml"))["sequentialize"]

    hist_size = params["hist_size"]

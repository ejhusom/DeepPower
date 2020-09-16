#!/usr/bin/env python3
# ============================================================================
# File:     config.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Configuration parameters.
# ============================================================================
from pathlib import Path

class Config:

    ASSETS_PATH = Path("./assets")
    DATA_PATH = ASSETS_PATH / "data"
    DATA_RESTRUCTURED_PATH = DATA_PATH / "restructured"
    DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
    DATA_SCALED_PATH = DATA_PATH / "scaled"
    MODELS_PATH = "../models"
    METRICS_PATH = "../metrics"
    METRICS_FILE_PATH = "../metrics/metrics.json"



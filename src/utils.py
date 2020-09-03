#!/usr/bin/env python3
# ============================================================================
# File:     utils.py
# Author:   Erik Johannes Husom
# Created:  2020-09-03
# ----------------------------------------------------------------------------
# Description:
# Utilities.
# ============================================================================
import os


def get_terminal_size():
    """Get size of terminal.

    Returns
    rows, columns : (int, int)
        Number of rows and columns in current terminal window.

    """

    with os.popen("stty size", "r") as f:
        termsize = f.read().split()

    return int(termsize[1]), int(termsize[0])



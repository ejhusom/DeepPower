#!/usr/bin/env python3
# ============================================================================
# File:     deeppower.py
# Author:   Erik Johannes Husom
# Created:  2020-08-25
# ----------------------------------------------------------------------------
# Description:
# Estimate power during workout using deep learning.
# ============================================================================
from preprocess import *
from model import *

import argparse
import pickle

RESULT_DIR = '../results/'
DATA_DIR = '../data/'


class DeepPower(Preprocess, NeuralTimeSeries):
    """Estimate power from breathing and heart rate, using deep learning.

    Parameters
    ----------
    parameter : float
      Description.



    Attributes
    ----------
    attribute : float
       Description.

    array[float]
       Description.


    Notes
    -----

    
    References
    ----------


    Example
    -------
    >>>

    """


    def __init__(
            self,
            data_file=DATA_DIR + "20200812-1809-merged.csv",
            time_id=time.strftime("%Y%m%d-%H%M%S"),
            hist_size = 5, # seconds
            net="cnn",
            train_split = 0.6,
            n_epochs=100,
            reverse_train_split=False
    ):


        self.data_file = data_file
        self.time_id = time_id
        self.hist_size = hist_size
        self.net = net
        self.train_split = train_split
        self.n_epochs = n_epochs
        
        Preprocess.__init__(self,
                hist_size=self.hist_size,
                train_split=self.train_split,
                data_file=self.data_file,
                reverse_train_split=reverse_train_split,
                time_id=self.time_id
        )


if __name__ == '__main__':
    np.random.seed(2020)

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-f", '--data_file', help='which data file to use', 
            default=DATA_DIR + "20200812-1809-merged.csv")
    parser.add_argument("-s", "--hist_size", type=int,
            help="""how many seconds of history to use for power estimation,
            default=5""", default=5)
    parser.add_argument('-n', "--net", 
            help="which network architectyre to use, default=cnn",
            default="cnn")
    parser.add_argument('--train_split', type=float,
            help='training/test ratio', default='0.6')
    parser.add_argument('-e', "--n_epochs", type=int, 
            help="number of epochs to run for NN, default=100", default=100)
    parser.add_argument('--reverse_train_split', help="""use first part of data
            set for testing and second part for training""",
            action='store_true')
    parser.add_argument('-l', '--load', help="loads model")
    parser.add_argument('--scaler', help='data scaler object')
    parser.add_argument('-t', '--train', help="trains model",
            action='store_true')
    parser.add_argument('-p', '--predict', help="predicts on test set",
            action='store_true')

    parser.add_argument('-f', '--features', nargs='+', default='',
            help="""
    Add extra features by writing the keyword after this flag. Available:
    - nan: No features available yet
    """)

    args = parser.parse_args()

    power_estimation = DeepPower(
            data_file=args.data_file,
            hist_size=args.hist_size,
            net=args.net,
            train_split=args.train_split,
            n_epochs=args.n_epochs,
            reverse_train_split=args-reverse_train_split
    )


    if args.scaler != None:
        power_estimation.set_scaler(args.scaler)

    power_estimation.preprocess(args.features)

    power_estimation.build_model()

    if args.load != None:
        if args.load.endswith('.h5'):
            inflow_analysis.set_model(args.load)


    if args.train:
        inflow_analysis.fit()

    if args.predict:
        inflow_analysis.predict()

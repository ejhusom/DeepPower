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
            hist_size = 50, # deciseconds
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

        self.title = ("""File: {}, hist_size: {}, net: {}, n_epochs: {}, added
        features: {}""".format(self.data_file, self.hist_size, self.net,
            self.n_epochs, self.added_features))
            
    def build_model(self):

        self.model = NeuralTimeSeries.__init__(
            self, self.X_train, self.y_train, self.X_test, self.y_test,
            self.n_epochs, self.net, self.time_id
        )
        print(self.model.summary())

        self.model_built = True

    def fit(self):
        """
        Fitting a model to the training data. This function is a wrapper
        function, in order to make it easy to swap between different methods.
        """

        if not self.model_built:
            self.build_model()
            print("yoyo")

        self._train_network()

    def predict(self, X_test=None, y_test=None):
        """Perform prediction using the trained model."""

        if X_test != None and y_test != None:
            self.X_test = X_test
            self.y_test = y_test

        self.y_pred = self.model.predict(self.X_test)

        # if self.y_scaler != None:
        #     self.y_test = self.y_scaler.inverse_transform(self.y_test)
        #     self.y_pred = self.y_scaler.inverse_transform(self.y_pred)
        #     print("Targets inverse transformed.")

    def plot_prediction(self):
        """
        Plot the prediction compared to the true targets.
        
        """

        # error_plot_average(self.y_test, self.y_pred, 168, self.time_id)

        plt.figure()
        plt.plot(self.y_test, label="true")
        plt.plot(self.y_pred, label="pred")

        # print("Pred MSE: {}".format(
        #     mean_squared_error(self.y_test, self.y_pred)
        # ))

        plt.legend()
        plt.title(self.title, wrap=True)
        plt.savefig(RESULT_DIR + self.time_id + "-pred.png")
        plt.show()

if __name__ == '__main__':
    np.random.seed(2020)

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-d", '--data_file', help='which data file to use', 
            default=DATA_DIR + "20200813-2012-merged.csv")
    parser.add_argument("-s", "--hist_size", type=int,
            help="""how many deciseconds of history to use for power estimation,
            default=5""", default=50)
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
            reverse_train_split=args.reverse_train_split
    )


    if args.scaler != None:
        power_estimation.set_scaler(args.scaler)

    power_estimation.preprocess(args.features)

    power_estimation.build_model()

    if args.load != None:
        if args.load.endswith('.h5'):
            power_estimation.set_model(args.load)


    if args.train:
        power_estimation.fit()

    if args.predict:
        power_estimation.predict()
        power_estimation.plot_prediction()

#!/usr/bin/env python3
# ============================================================================
# File:     deeppower.py
# Author:   Erik Johannes Husom
# Created:  2020-08-25
# ----------------------------------------------------------------------------
# Description:
# Estimate power during workout using deep learning.
# ============================================================================
import argparse
import os
import pickle

try:
    import gnuplotlib as gp
except:
    pass

from preprocess import *
from model import *


class DeepPower(Preprocess, NeuralTimeSeries):
    """Estimate power from breathing and heart rate, using deep learning.

    Parameters
    ----------


    Attributes
    ----------

    """


    def __init__(self,
            data_file="../data/20200812-1809-merged.csv",
            hist_size=1000, 
            train_split=0.6, 
            scale=True,
            reverse_train_split=False, 
            verbose=False,
            net="cnn",
            n_epochs=100,
            time_id=time.strftime("%Y%m%d-%H%M%S"),
    ):

        self.net = net
        self.n_epochs = n_epochs

        Preprocess.__init__(self,
                data_file=data_file,
                hist_size=hist_size,
                train_split=train_split,
                scale=scale,
                reverse_train_split=reverse_train_split,
                verbose=verbose,
                time_id=time_id
        )

        self.title = (
            """File: {}, hist_size: {}, net: {}, n_epochs: {}, 
            added feats.: {}""".format(
                self.data_file, self.hist_size, self.net, self.n_epochs,
                self.added_features
            )
        )

            
    def build_model(self):
        """Build the model."""

        try:
            NeuralTimeSeries.__init__(
                self, self.X_train, self.y_train, self.X_test, self.y_test,
                self.n_epochs, self.net, self.time_id
            )
        except:
            raise AttributeError("Data is not preprocessed.")


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

    def plot_prediction(self, include_input=True):
        """
        Plot the prediction compared to the true targets.
        """

        # error_plot_average(self.y_test, self.y_pred, 168, self.time_id)

        plt.figure()

        plt.plot(self.y_test, label="true")
        plt.plot(self.y_pred, label="pred")

        if include_input:
            for i in range(self.X_test_pre_seq.shape[1]):
                # plt.plot(self.df.iloc[:,i], label=self.input_columns[i])
                plt.plot(self.X_test_pre_seq[:,i]*250, label=self.input_columns[i])

        plt.legend()
        plt.title(self.title, wrap=True)
        plt.autoscale()
        plt.savefig(self.result_dir + self.time_id + "-pred.png")
        plt.show()

    def plot_prediction_plotly(self, include_input=True):
        """
        Plot the prediction compared to the true targets, using plotly.
        """

        x_len = len(self.y_test.flatten())
        x = np.linspace(0,x_len-1,x_len)

        fig = go.Figure()
        config = dict({"scrollZoom": True})

        fig.add_trace(go.Scatter(
            x=x, y=self.y_test.flatten(), name="true"))
        fig.add_trace(go.Scatter(
            x=x, y=self.y_pred.flatten(), name="pred"))

        if include_input:
            for i in range(self.X_test_pre_seq.shape[1]):
                fig.add_trace(go.Scatter(
                    x=x, y=self.X_test_pre_seq[:,i]*250,
                    name=self.input_columns[i]))

        fig.show(config=config)

    def plot_prediction_gp(self):
        """
        Plot the prediction compared to the true targets, using gnuplotlib.
        """

        with os.popen("stty size", "r") as f:
            termsize = f.read().split()
        termsize[0], termsize[1] = int(termsize[1]), int(termsize[0])

        y_true = self.y_test.flatten()
        y_pred = self.y_pred.flatten()

        x_len = len(y_true)
        x = np.linspace(0,x_len-1,x_len)

        gp.plot((x, y_true, dict(legend="true")),
                (x, y_pred, dict(legend="pred")),
                unset="grid",
                terminal="dumb {} {}".format(termsize[0], termsize[1]))

        # import plotext.plot as plx
        # plx.plot(x, y_true, line=True)
        # plx.plot(x, y_pred, line=True)
        # plx.show()

        # import termplotlib as tpl
        # fig = tpl.figure()
        # fig.plot(x, y_true)
        # fig.plot(x, y_pred)
        # fig.show()


if __name__ == '__main__':
    np.random.seed(2020)
    time_id = time.strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-v', '--verbose', action="store_true",
            help="print what the program does")
    parser.add_argument('-g', '--gnuplotlib', action="store_true",
            help="use gnuplotlib for plotting")
    parser.add_argument('--plotly', action="store_true",
            help="use plotly for plotting")

    # PREPROCESSING ARGUMENT
    parser.add_argument("-d", '--data_file', 
            default="../data/20200813-2012-merged.csv",
            help='which data file to use')
    parser.add_argument("-s", "--hist_size", type=int, default=50,
            help="""how many deciseconds of history to use for power estimation,
            default=5""")
    parser.add_argument('--train_split', type=float, default="0.6",
            help='training/test ratio')
    parser.add_argument('--reverse_train_split', action="store_true",
            help="""use first part of data set for testing and second part for
            training""")
    parser.add_argument('-f', '--features', nargs='+', default='',
            help="""
    Add extra features by writing the keyword after this flag. Available:
    - nan: No features available yet
    """)

    # MODEL ARGUMENTS
    parser.add_argument('-n', "--net", default="cnn",
            help="which network architectyre to use, default=cnn")
    parser.add_argument('-e', "--n_epochs", type=int, default=100,
            help="number of epochs to run for NN, default=100")
    parser.add_argument('-t', '--train', action="store_true",
            help="trains model")
    parser.add_argument('-p', '--predict', action="store_true",
            help="predicts on test set")

    # LOAD MODEL ARGUMENTS
    parser.add_argument('-m', '--model', help="loads pretrained model")
    parser.add_argument('--scaler', help='data scaler object')

    args = parser.parse_args()

    power_estimation = DeepPower(
            data_file=args.data_file,
            hist_size=args.hist_size,
            train_split=args.train_split,
            reverse_train_split=args.reverse_train_split,
            net=args.net,
            n_epochs=args.n_epochs,
            verbose=args.verbose,
            time_id=time_id
    )

    power_estimation.preprocess(args.features)
    power_estimation.build_model()

    if args.model != None:
        if args.model.endswith('.h5'):
            power_estimation.set_model(args.model)
        else:
            raise ValueError("Model does not have correct extension.")

        if args.scaler != None:
            power_estimation.set_scaler(args.scaler)
        else:
            raise Exception("To load pretrained model, scaler must be given.")
            sys.exit(1)

    if args.train:
        power_estimation.fit()

    if args.predict:
        power_estimation.predict()
        if args.plotly:
            power_estimation.plot_prediction_plotly()
        elif args.gnuplotlib:
            power_estimation.plot_prediction_gp()
        else:
            power_estimation.plot_prediction()


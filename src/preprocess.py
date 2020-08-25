#!/usr/bin/env python3
# ============================================================================
# File:     preprocess.py
# Created:  2020-08-24
# Author:   Erik Johannes Husom
# ----------------------------------------------------------------------------
# Description: Preprocessing workout data.
# ============================================================================
import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [5.0, 3.0]
# plt.rcParams['figure.dpi'] = 300

import datetime 
import numpy as np
import pandas as pd
import pickle
import string
import sys
import time
from scipy.fftpack import fft, ifft
 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

RESULT_DIR = '../results/'
DATA_DIR = '../data/'

def split_sequences(sequences, hist_size, target_col=0, n_steps_out=1):
    """Split data sequence into samples with matching input and targets.

    Parameters
    ----------
    sequences : array
        The matrix containing the sequences, with the targets in the first
        column.
    hist_size : int
        Number of time steps to include in each sample, i.e. how much history
        should be matched with a given target.
    n_steps_out : int
        Number of output steps.
    
    Returns
    -------
    X : array
        The input samples.
    y : array
        The targets.
    
    """
    X, y = list(), list()
	
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + hist_size
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # Delete target col from sequences, which leaves input matrix
        seq_x = np.delete(sequences[i:end_ix], target_col, axis=1)
        # Extract targets from sequences
        seq_y = sequences[end_ix:out_end_ix, target_col]

        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    return X, y

def merge_time_series_and_added_features(X):
    """
    Reverse the operation done on input matrix X by
    split_time_series_and_added_features, but flattening the time series
    data.

    Parameters
    ----------
    X : list
        This must be a list of two elements:
        1. 2D-array of shape [hist_size, num_features], which contains the time
           series data.
        2. 1D-array of shape [num_added_features], which contains the added
           features.

    Return
    ------
    result : array
        A 2D-array, where each row contains the flattened time series data
        along with the added features, such that each row contains the input
        data needed to make one prediction.

    """
    
    if isinstance(X, list) and (len(X) == 2):

        result = list()

        for i in range(len(X[0])):
            row = np.concatenate([
                X[0][i].reshape(-1),
                X[1][i]
            ])
            result.append(row)

        return np.array(result)

    else:
        raise TypeError("X must be a list of two elements.")


def scale_data(train_data, val_data, scaler_type='minmax'):
    """Scale train and test data.

    Parameters
    ----------
    train_data : array
        Train data to be scaled. Used as scale reference for test data.
    val_data : array
        Test data too be scaled, with train scaling as reference.
    scaler_type : str, default='standard'
        Options: 'standard, 'minmax'.
        Specifies whether to use sklearn's StandardScaler or MinMaxScaler.

    Returns
    -------
    train_data : array
        Scaled train data.
    val_data : array
        Scaled test data.
    sc : scikit-learn scaler
        The scaler object that is used.

    """

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        print('Scaler must be "standard" or "minmax"!')
        return None

    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    return train_data, val_data, scaler

def split_time_series_and_added_features(X, input_columns, added_features):
    """
    Take the result from split_sequences(), remove weather forecast for all time
    steps (in each sample) except the latest one, and put the latest forecast
    into a separate array. The goal is to have one input matrix for the
    historical observations, which will be given to a CNN, and an array for
    the weather forecast, which will be fed into a dense NN. The networks will
    later be combined. The purpose of this is to remove the outdated forecast
    for each sample, in order to make the input less complex.

    Parameters
    ----------
    X : list/array of arrays
        The input matrix produced by split_sequences().
    input_columns : list of strings
        An array/list that contains the names of the columns in the input
        matrix. This is used to sort out which columns should be a part of the
        historic data, and which data that belongs to the forecast. The sorting
        is done based on that the name of the forecast columns start with a
        digit (the coordinates), while the other columns do not.
    added_features : list
        A list of the features that are added to the raw data. These features
        will not be included in the history window, but will be appended to the
        array of forecast values.
    
    Returns
    -------
    X_hist : list of arrays
        The input matrix containing historical observations, with the weather
        forecast removed.
    X_forecast : list of arrays
        Input matrix containing the latest weather forecast for each sample.
    
    """

    X_hist, X_added = list(), list()
    hist_idcs = []
    added_idcs = []

    for i in range(len(input_columns)):
        if input_columns[i] in added_features:
            added_idcs.append(i)
        else:
            hist_idcs.append(i)

	
    for i in range(len(X)):
        X_hist.append(X[i][:,hist_idcs])
        X_added.append(X[i][-1,added_idcs])

    return [np.array(X_hist), np.array(X_added)]



class Preprocess():

    def __init__(self, hist_size=1000, train_split=0.6, scale=True,
            data_file=DATA_DIR + "20200812-1809-merged.csv",
            reverse_train_split=False, time_id=time.strftime("%Y%m%d-%H%M%S") 
        ):
        """
        Load and preprocess data set.
        
        Parameters
        ----------
        date : string
            The last day of the desired data set.
        hist_size : int, default=1000
            How many past time steps should be used for prediction.
        train_split : float, (0,1)
            How much data to use for training (the rest will be used for testing.)
        scale : bool, default=True
            Whether to scale the data set or not.
        data_file : string, default="DfEtna.csv"
            What file to get the data from.
        reverse_train_split : boolean, default=False
            Whether to use to first part of the dataset as test and the second
            part for training (if set to True). If set to False, it uses the
            first part for training and the second part for testing.

        """

        self.data_file = data_file
        self.train_split = train_split
        self.hist_size = hist_size
        self.scale = scale
        self.reverse_train_split = reverse_train_split
        self.time_id = time_id

        self.scaler_loaded = False
        self.added_features = []

        # Get input matrix from file
        self.df = pd.read_csv(self.data_file, index_col=0)
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        self.index = self.df.index
        print(self.df)
        print("Data file loaded: {}".format(self.data_file))
        print("Length of data set: {}".format(len(self.df)))


        self.target_col_name = "power"


    def preprocess(self, features = [], diff=False):


        del self.df["time"]
        del self.df["calories"]

        self.add_features(features)
        # Save the names of the input columns
        # self.input_columns = self.df.columns
        # input_columns_df = pd.DataFrame(self.input_columns)
        # input_columns_df.to_csv(RESULT_DIR + self.time_id +
        #         "-input_columns.csv")


        self.data = self.df.to_numpy()

        self._split_train_test()
        # self._scale_data()

        # Save data for inspection of preprocessed data set
        self.df.to_csv(RESULT_DIR + "tmp_df.csv")
        np.savetxt(RESULT_DIR + "tmp_X_train.csv", self.X_train, delimiter=",")

        self._split_sequences()


        plt.plot(self.X_train[0])
        plt.show()

        self.n_features = self.X_train.shape[-1]
        # self._create_feature_dict()

        # Save test targets for inspection
        np.savetxt(RESULT_DIR + "tmp_y_test.csv", self.y_test, delimiter=",")


    def add_features(self, features):
        """
        This function adds features to the input data, based on the arguments
        given in the features-list.

        Parameters
        ----------
        features : list
            A list containing keywords specifying which features to add.
            
        """
        # TODO: Remove the need for the variable t_vt, since it will need to be
        # changed if Sortungen temperature and/or precipitation is added to
        # inputs. Better to use variable t_col (name of column), such that the
        # variable do not need to be changed.
        pass

    def add_feature(self, name, feature_col, add_to_hist_matrix=False):
        """
        Adding a feature to the data set. The name is 'registered' into one of
        two lists, in order to keep track of what features that are added to
        the raw data.

        The differences between the two lists are as follows: When using the
        CNNDense submethod, the list self.added_features consists of features
        that will be sent to the Dense-part of the network. If the parameter
        'add_to_hist_matrix' is set to True, the feature will be registered in
        the other list, self.added_hist_features, which will be sent to the
        CNN-part of the network when using CNNDense, together with the
        observation history of the data set. The separation of the two lists
        only matter when using the CNNDense submethod.

        The point of this function is to make a consistent treatment of added
        features and reducing the number of lines required to add a feature.

        Parameters
        ----------
        name : string
            What to call the new feature.
        feature_col : array-like
            The actual data to add to the input matrix.
        add_to_hist_matrix : boolean, default=False
            Whether to use the feature in as historical data, meaning that data
            points from previous time steps also will be included in the input
            matrix. If set to True, only the current data point will be used as
            input.

        """

        self.df[name] = feature_col

        if add_to_hist_matrix:
            self.added_hist_features.append(name)
        else:
            self.added_features.append(name)

        print('Feature added: {}'.format(name))

    def _split_sequences(self):
        """Wrapper function for splitting the input data into sequences. The
        point of this function is to accomodate for the possibility of
        splitting the data set into seasons, and fitting a model for each
        season.

        """

        self.X_train_pre_seq = self.X_train.copy()
        self.X_test_pre_seq = self.X_test.copy()
        # Combine data 
        self.train_data = np.hstack((self.y_train, self.X_train))
        self.test_data = np.hstack((self.y_test, self.X_test))

        self.X_train, self.y_train = split_sequences(
            self.train_data, self.hist_size, target_col=self.power_col
        )
        self.X_test, self.y_test = split_sequences(
            self.test_data, self.hist_size
        )

    def _split_train_test(self, test_equals_train=True):
        """
        Splitting the data set into training and test set, based on the
        train_split ratio.
        """

        self.train_hours = int(self.data.shape[0]*self.train_split)

        self.train_data = self.data[:self.train_hours,:]
        self.train_indeces = self.index[:self.train_hours]
        self.test_data = self.data[self.train_hours:,:]
        self.test_indeces = self.index[self.train_hours:]

        if test_equals_train:
            self.test_data = self.train_data
            self.test_indeces = self.train_indeces



        # Split data in inputs and targets
        self.X_train = self.train_data
        self.X_test = self.test_data
        self.y_train = self.train_data[:,0].reshape(-1,1)
        self.y_test = self.test_data[:,0].reshape(-1,1)

    def _scale_data(self, scaler_type='minmax'):
        """
        Scaling the input data.

        Default behaviour is to create a scaler based on the training data, and
        then scale the test set with this scaler. If a scaler object has been
        loaded, by using the function set_scaler(), then the dataset will be
        scaled using the loaded scaler.

        Parameters
        ----------
        scaler_type : string, default='minmax'
            What type of scaling to perform, Not applicable if a scaler object
            is loaded.

        """

        if self.scale:

            if self.scaler_loaded:

                try:
                    self.X_train = self.X_scaler.transform(self.X_train)
                except:
                    pass

                self.X_test = self.X_scaler.transform(self.X_test)
                print("Loaded scaler used to scale data.")


            else:
                if scaler_type == 'standard':
                    self.X_scaler = StandardScaler()
                elif scaler_type == 'robust':
                    self.X_scaler = RobustScaler()
                else:
                    self.X_scaler = MinMaxScaler()

                self.X_train = self.X_scaler.fit_transform(self.X_train)
                self.X_test = self.X_scaler.transform(self.X_test)

                # Save the scaler in order to reuse it on other test sets
                pickle.dump(self.X_scaler, open(RESULT_DIR + self.time_id +
                    '-scaler.pkl', 'wb'))

                print("Data scaled and scaler saved.")


    def set_scaler(self, scaler_file):
        """
        Loading a saved scaler object from a previous data preprocessing.
        Useful when testing a model on new data.
        """

        self.X_scaler = pickle.load(open(scaler_file, 'rb'))
        self.scaler_loaded = True

        print("Scaler loaded : {}".format(scaler_file))


    def _create_feature_dict(self):
        """
        Create an array that contains the name of the features in the exact
        order as they appear in the tabular version of the input matrix, i.e.
        the input matrix used for boosting etc, not the sliding window version
        used for neural networks. The function creates an numpy array, and not
        a Python dictionary, in order to make it easier to extract names using
        array of indeces.
        """

        self.feature_dict = []

        for i in range(self.hist_size):
            # for j in range(self.n_features):
            for j in range(5):
                value = (
                    self.input_columns[j] +
                    '_Tminus{}'.format(abs(i-self.hist_size))
                )
                self.feature_dict.append(value)
        for i in range(5, self.n_features):
            self.feature_dict.append(self.input_columns[i])

        # pp = pprint.PrettyPrinter()
        # pp.pprint(self.feature_dict)
        # print(self.feature_dict)
        self.feature_dict = np.array(self.feature_dict)

    def augment_data(self, thresh=20):
        """
        Augment data from certain periods.

        Parameters
        ----------
        thresh : int
            If a target vector contains a value above this threshold, the data
            point will be duplicated.

        """

        print('Augmenting data...')
        original_num_points = self.y_train.shape[0]

        for i in range(len(self.X_train[0])):
            if np.max(self.y_train[i,:]) > thresh:
                self.y_train = np.vstack((self.y_train, self.y_train[i,:]))
                self.X_train[0] = np.concatenate((
                    self.X_train[0], np.array([self.X_train[0][i]])))
                self.X_train[1] = np.concatenate((
                    self.X_train[1], np.array([self.X_train[1][i]])))
                self.X_train[2] = np.concatenate((
                    self.X_train[2], np.array([self.X_train[2][i]])))

        new_num_points = self.y_train.shape[0]

        print('Original number of points: {}.'.format(original_num_points))
        print('New number of points: {}.'.format(new_num_points))
        print('Data augmented with {} points.'.format(
            new_num_points-original_num_points
        ))


if __name__ == '__main__':

    data = Preprocess()
    data.preprocess()

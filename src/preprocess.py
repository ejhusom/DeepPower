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

def split_sequences(sequences, hist_size, n_steps_out=1):
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
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, 1:]
        seq_y = sequences[end_ix:out_end_ix, 0]
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

def get_polynomials(inflow):
    y = np.linspace(0, 10, len(inflow))
    first_order_2_point = np.zeros(np.shape(inflow))
    first_order_3_point = np.zeros(np.shape(inflow))
    second_order = np.zeros(np.shape(inflow))
    for i in range(5, len(inflow)):
        f2 = np.polyfit(y[i-2:i].copy(), inflow[i-2:i].copy(), 1)
        f3 = np.polyfit(y[i-3:i].copy(), inflow[i-3:i].copy(), 1)
        s = np.polyfit(y[i-3:i].copy(), inflow[i-3:i].copy(), 2)
        F2 = np.poly1d(f2)
        F3 = np.poly1d(f3)
        S = np.poly1d(s)
        first_order_2_point[i] = F2(y[i])
        first_order_3_point[i] = F3(y[i])
        second_order[i] = S(y[i])
    return first_order_2_point, first_order_3_point, second_order

def get_longrun_polynomial(inflow, hours_hist, hours_ahead, degree):

    y = np.linspace(0, 10, len(inflow))
    polynom = np.repeat(np.array([y]), hours_ahead, axis=0)

    j=hours_hist
    for i in range(hours_hist, len(inflow)-hours_ahead):
        z = np.polyfit(y[i-hours_hist:i].copy(), inflow[i-hours_hist:i].copy(), degree)
        p = np.poly1d(z)
        polynom[:,j] = p(y[i:i+hours_ahead])
        j+=1
    return polynom


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


    def preprocess(self, features = [], diff=False):

        self.add_features(features)

        # Save the names of the input columns
        self.input_columns = self.df.columns
        input_columns_df = pd.DataFrame(self.input_columns)
        input_columns_df.to_csv(RESULT_DIR + self.time_id +
                "-input_columns.csv")


        self.data = self.df.to_numpy()

        self._split_train_test()
        self._scale_data()

        # Save data for inspection of preprocessed data set
        self.df.to_csv(RESULT_DIR + "tmp_df.csv")
        np.savetxt(RESULT_DIR + "tmp_X_train.csv", self.X_train, delimiter=",")

        self._split_sequences()

        self.n_features = self.X_train.shape[-1]
        self._create_feature_dict()

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
        t_vt = 2 # temperature Vest-Torpa index, if Sortungen is not included

        t_col = 'Vest-Torpa_lufttemperatur'
        p_col = 'Vest-Torpa_nedbor_time'
        temp_rolling_mean_window = 24
        temp_rolling_extrema_window = 48
        prec_rolling_sum_window = 96#48
        prec_sum_threshold = 20
        inflow_diff_shift = 12#96


        if 'inflow_diff' in features:
            """The difference in inflow with a given shift."""

            self.add_feature(
                    'inflow_diff_{}'.format(inflow_diff_shift),
                    self.df.iloc[:,1].diff(inflow_diff_shift)
            )

        if 'month' in features:
            self.add_feature('month', self.dates.month)

        if 'season' in features:
            """
            Split the year into three seasons (winter, spring, summer), and add
            this as a categorical variable.
            """

            self.df["month"] = self.dates.month
            """Add season : 0=winter, 1=spring, 2 = summer"""
            self.df["season"] = np.minimum([(month%10 + 1)//4 for month in self.df["month"]], 2)
            self.added_features.append('season')
            del self.df['month']

        if 'poly' in features:
            """Adds polynomial approximation of last inflow points, using leat squares method"""
            self.add_polynomials()

        if 'longrun' in features or 'long_run' in features:
            """Adds polynomial approximation of longer period of inflow points, using leat squares method"""
            self.add_longrun()

        if 'hour' in features:
            """Cyclic encoding of hour of the day."""

            self.add_feature(
                    'hour_sin', np.sin(2*np.pi*self.dates.hour/24)
            )
            self.add_feature(
                    'hour_cos', np.cos(2*np.pi*self.dates.hour/24)
            )

        if 'day' in features:
            """Cyclic encoding of day of the year."""

            self.add_feature(
                    'day_sin', np.sin(2*np.pi*self.dates.dayofyear/365)
            )
            self.add_feature(
                    'day_cos', np.cos(2*np.pi*self.dates.dayofyear/365)
            )

        if 'current_inflow' in features:
            self.add_feature('inflow_1H_ago', self.df.iloc[:,1])

        if '10_days' in features:
            """
            Cyclic encoding of 10 day period. The inflow difference (with 1
            hour shift) seems to be having a peak frequency at a period of 10
            days.
            """
            
            unixtime = self.dates.astype(np.int64)

            secs_in_day = 60*60*24
            secs_in_10_days = 10*secs_in_day

            self.add_feature(
                    '10_day_sin', np.sin(2*np.pi*unixtime/secs_in_10_days)
            )
            self.add_feature(
                    '10_day_cos', np.cos(2*np.pi*unixtime/secs_in_10_days)
            )


        if 'fft_prec' in features or 'fft_precip' in features:
            """Fast fourier transform of precipitation"""
            self.add_fft_historical(20, 'precip', 3)

        if 'fft_temp' in features:
            """Fast fourier transform of temperature"""
            self.add_fft_historical(20, 'temp', t_vt)

        if 'fft_inflow' in features:
            """Fast fourier transform of inflow"""
            self.add_fft_historical(20, 'inflow', 1)

        if 'fft_forecast' in features:
            """Fast fourier transform of forecast"""
            self.add_fft_forecast()

        if 'temp_forecast_change' in features:
            """The change in forecast values form previos forecast (6 hours ago), on temperature forecast"""
            self.add_temp_forecast_change()

        if 'prec_forecast_change' in features:
            """The change in forecast values form previos forecast (6 hours ago), on precipitation forecast"""
            self.add_prec_forecast_change()

        if 'days_since_prec' in features:
            """Number of days since the precipitation amount was above 5 mm"""
            self.add_days_since_xmm_prec(5)


        if 'temp_mean' in features:
            """Rolling mean of temperature."""

            self.add_feature(
                    'temp_rolling_mean_vesttorpa_{}'.format(
                        temp_rolling_mean_window
                    ),
                    self.df.iloc[:,2].rolling(temp_rolling_mean_window).mean()
            )
            
        if 'temp_mean_bool' in features:
            """Boolean of whether the rolling mean is plus or minus degrees."""
            
            self.add_feature(
                    'temp_rolling_mean_boolean_vesttorpa_{}'.format(
                        temp_rolling_mean_window
                    ),
                    (self.df[t_col].rolling(
                        temp_rolling_mean_window
                    ).mean() > 0).astype(int),
            )

        if 'temp_range' in features:
            """Rolling temperature range."""

            # Rolling temperature minimum
            temp_rolling_min_vesttorpa = self.df[t_col].rolling(
                    temp_rolling_extrema_window
                ).min()

            # Rolling temperature maximum
            temp_rolling_max_vesttorpa = self.df[t_col].rolling(
                    temp_rolling_extrema_window
                ).max()

            # Rolling temperature range
            temp_rolling_range_vesttorpa = (
                    temp_rolling_max_vesttorpa - temp_rolling_min_vesttorpa
            )
            self.add_feature(
                    'temp_rolling_range_vesttorpa_{}'.format(
                        temp_rolling_extrema_window
                    ),
                    temp_rolling_range_vesttorpa
            )

        if 'temp_range_mean' in features:
            """Mean rolling temperature range."""

            # Rolling temperature minimum
            temp_rolling_min_vesttorpa = self.df[t_col].rolling(
                    temp_rolling_extrema_window
                ).min()

            # Rolling temperature maximum
            temp_rolling_max_vesttorpa = self.df[t_col].rolling(
                    temp_rolling_extrema_window
                ).max()

            # Rolling temperature range
            temp_rolling_range_vesttorpa = (
                    temp_rolling_max_vesttorpa - temp_rolling_min_vesttorpa
            )
            self.add_feature(
                    'temp_rolling_range_vesttorpa_{}'.format(
                        temp_rolling_extrema_window
                    ),
                    temp_rolling_range_vesttorpa.rolling(240).mean()
            )

        if 'temp_min_bool' in features:
            """Boolean of whether the rolling minimum is plus or minus degrees."""

            self.add_feature(
                    'temp_rolling_min_boolean_vesttorpa_{}'.format(
                        temp_rolling_extrema_window
                    ),
                    (self.df[t_col].rolling(
                        temp_rolling_extrema_window
                    ).min() > 0).astype(int)
            )

        if 'temp_max' in features:
            """Rolling max value of temperature"""
            self.add_feature('temp_rolling_max_vesttorpa_{}'.format(temp_rolling_extrema_window), 
                    (self.df[t_col].rolling(temp_rolling_extrema_window).max()))

        if 'temp_min' in features:
            """Rolling min value of temperature"""
            self.add_feature('temp_rolling_min_vesttorpa_{}'.format(temp_rolling_mean_window), 
                    (self.df[t_col].rolling(temp_rolling_mean_window).min()))

        if 'prec_sum' in features:
            """Rolling sum of precipitation."""

            self.add_feature(
                    'prec_rolling_sum_vesttorpa_{}'.format(
                        prec_rolling_sum_window
                    ),
                    self.df[p_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )

        if 'days_since_inflow' in features:
            """Number of days since the inflow was over 17"""
            self.add_days_since_x_inflow(15)

        if 'prec_sum_forecast' in features:
            """Rolling sum of precipitation forecast."""

            forecast_hour = 24
            forecast_col = self.get_forecast_column_name(
                    hour=forecast_hour, forecast_type='prec')

            prec_sum_forecast = (
                    self.df[forecast_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )
            self.add_feature(
                    'prec_forecast_{}H_rolling_sum_{}'.format(
                        forecast_hour, prec_rolling_sum_window
                    ),
                    prec_sum_forecast
            )

        if 'prec_sum_thresh' in features:
            """
            Boolean feature, that indicates whether the rolling sum of
            precipitation is above a given threshold.
            """

            prec_rolling_sum = (
                    self.df[p_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )
            prec_sum_thresh = (prec_rolling_sum > prec_sum_threshold).astype(int)

            self.add_feature(
                    'prec_sum_thresh_{}win_{}thresh'.format(
                        prec_rolling_sum_window,
                        prec_sum_threshold
                    ),
                    prec_sum_thresh
            )

        if 'days_since_prec_sum' in features:
            """days since the rolling sum of precipitation was above x mm"""
            x = 30

            prec_rolling_sum = (
                    self.df[p_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )
            prec_rolling_sum = prec_rolling_sum.to_numpy().copy()
            prec_rolling_sum[prec_rolling_sum<x] = 0
            days_since = np.array([])
            days = 0
            for i in range(len(prec_rolling_sum)):
                if ((prec_rolling_sum[i] != 0) or (i == len(prec_rolling_sum)-1)):
                    days_since = np.append(days_since, np.arange(0, days + 1, 1))
                    days = 0
                else:
                    days+=1
            days_since[days_since>1400] = 1400
            self.df['days_since_prec_sum_over_{}mm'.format(x)] = days_since
            self.added_features.append('days_since_prec_sum_over_{}mm'.format(x))
            print('Feature added: Days since {}mm prec sum'.format(x))

        if 'clausius' in features:
            """August–Roche–Magnus formula for vapour pressure, hoping to impove the models take on soil moisture"""
            temp = self.df.iloc[:,t_vt].to_numpy().copy() + 273
            e = 6.194*np.exp((17.625*temp)/(temp+243.04))
            self.df['clausius'] = e
            self.added_features.append('clausius')
            print('Feature added: clausius')

        if 'prec_sum_thresh_forecast' in features:
            """
            Boolean feature, that indicates whether the rolling sum of
            precipitation forecast is above a given threshold.
            """

            forecast_hour = 24
            forecast_col = self.get_forecast_column_name(
                    hour=forecast_hour, forecast_type='prec')

            prec_rolling_sum = (
                    self.df[forecast_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )
            prec_sum_thresh = (prec_rolling_sum > prec_sum_threshold).astype(int)

            self.add_feature(
                    'prec_sum_thresh_forecast_{}H_{}win_{}thresh'.format(
                        forecast_hour,
                        prec_rolling_sum_window,
                        prec_sum_threshold
                    ),
                    prec_sum_thresh
            )
            
        if 'prec_sum_thresh_forecast_summer' in features:
            """
            Boolean feature, that indicates whether the rolling sum of
            precipitation forecast is above a given threshold, BUT only in the
            summer.
            """
            raise NotImplementedError

            forecast_hour = 24
            forecast_col = self.get_forecast_column_name(
                    hour=forecast_hour, forecast_type='prec')

            prec_rolling_sum = (
                    self.df[forecast_col].rolling(
                        prec_rolling_sum_window
                    ).sum()
            )
            prec_sum_thresh = (prec_rolling_sum > prec_sum_threshold).astype(int)

            start = 6
            stop = 10
    
            summer_mask = (self.dates.month >= start) & (self.dates.month < stop)

            print(summer_mask)

            prec_sum_thresh = prec_sum_thresh[summer_mask]
            # prec_sum_thresh = pd.DataFrame(
            #         prec_sum_thresh,
            #         index=self.dates
            # )
            # plt.plot(prec_sum_thresh)
            plt.plot(self.dates, prec_sum_thresh)
            plt.show()
            print(prec_sum_thresh)

        self.df = self.df.dropna()
        self.df.reset_index(inplace=True, drop=True)

    def add_rolling_features(self, features):
        """
        Function kept for backward compatibility. Replaced by add_features.
        """

        self.add_features(features)
    
    def add_fft_historical(self, num, var_name, colnr, min_v = 4, max_v = 4):
        # y = np.linspace(0, 10, len(inflow))
        inflow = self.df.iloc[:,1].to_numpy()
        var = self.df.iloc[:,colnr].to_numpy()

        # var_fourier = np.zeros((num, len(inflow)))#, dtype = complex)
        var_fourier = np.zeros((num, len(inflow)))#min_v+max_v, len(inflow)))#, dtype = complex)

        for i in range(num, len(inflow)):
            # f = fft(var[i-num:i])
            # var_fourier[:,i] = np.array([list(list(np.sort(f)[0:max_v]) + list(np.sort(f)[-min_v:]))])
            var_fourier[:,i] = fft(var[i-num:i])
        
        for i in range(num): #min_v+max_v):
            self.df['fourrier_' + str(var_name) + '{}'.format(i)] = var_fourier[i]
            self.added_features.append('fourrier_ ' + str(var_name) + '{}'.format(i))

        print('Feature added: fft {}'.format(var_name))

    def add_fft_forecast(self):
        forecast_temp = np.array([fft(x) for x in self.df.iloc[:,4:40].to_numpy()], dtype = float)
        forecast_precip = np.array([fft(x) for x in self.df.iloc[:,41:77].to_numpy()], dtype = float)
        for i in range(0, len(forecast_temp[0])):
            self.df['fourrier_temp_forecast_{}'.format(i)] = forecast_temp[:,i]
            self.added_features.append('fourrier_temp_forecast{}'.format(i))
        for i in range(0, len(forecast_temp[0])):
            self.df['fourrier_precip_forecast{}'.format(i)] = forecast_precip[:,i]
            self.added_features.append('fourrier_precip_forecast{}'.format(i))
        print('Feature added: fft forecast')
    

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


    def _split_train_test(self):
        """
        Splitting the data set into training and test set, based on the
        train_split ratio.
        """

        if self.reverse_train_split:
            self.test_hours = int(self.data.shape[0]*(1-self.train_split))
            
            self.train_data = self.data[self.test_hours:,:]
            self.train_indeces = self.index[self.test_hours:]
            self.test_data = self.data[:self.test_hours,:]
            self.test_indeces = self.index[:self.test_hours]

            print('Reversing train split.')
        else:
            self.train_hours = int(self.data.shape[0]*self.train_split)

            self.train_data = self.data[:self.train_hours,:]
            self.train_indeces = self.index[:self.train_hours]
            self.test_data = self.data[self.train_hours:,:]
            self.test_indeces = self.index[self.train_hours:]



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
            self.train_data, self.hist_size
        )
        self.X_test, self.y_test = split_sequences(
            self.test_data, self.hist_size
        )

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

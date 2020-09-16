#!/usr/bin/env python3
# ============================================================================
# File:     featurize.py
# Author:   Erik Johannes Husom
# Created:  2020-09-16
# ----------------------------------------------------------------------------
# Description:
# Add engineered features to dataset.
# ============================================================================
import numpy as np
import pandas as pd


class Featurize():

    def __init__(self, 
            filename="../data/20200812-1809-merged.csv",
            hist_size=1000, 
            train_split=0.6, 
            scale=True,
            reverse_train_split=False, 
            verbose=False,
            target_name="power",
            time_id=time.strftime("%Y%m%d-%H%M%S")
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
        filename : string, default="DfEtna.csv"
            What file to get the data from.
        reverse_train_split : boolean, default=False
            Whether to use to first part of the dataset as test and the second
            part for training (if set to True). If set to False, it uses the
            first part for training and the second part for testing.
        verbose : boolean, default=False
            Printing what the program does.

        """

        params = yaml.safe_load(open("params.yaml"))["featurize"]

        self.filename = filename
        self.train_split = train_split
        self.hist_size = hist_size
        self.scale = scale
        self.reverse_train_split = reverse_train_split
        self.time_id = time_id
        self.verbose = verbose

        self.preprocessed = False
        self.scaler_loaded = False
        self.added_features = []
        self.target_name = target_name
        self.result_dir = "../results/" + time_id + "/"
        os.makedirs(self.result_dir)

    
    def preprocess(self, features = [], remove_features = []):
        
        self.df, self.index = read_csv(
                self.filename, 
                delete_columns=["time", "calories"] + remove_features,
                verbose=self.verbose
        )
        print(self.df)

        # Move target column to the beginning of dataframe
        self.df = move_column(self.df, self.target_name, 0)

        if self.verbose:
            print_horizontal_line()
            print("DATAFRAME BEFORE FEATURE ENGINEERING")
            print(self.df)


        # self.add_features(features)

        # Save the names of the input columns
        self.input_columns = self.df.columns
        input_columns_df = pd.DataFrame(self.input_columns)
        input_columns_df.to_csv(self.result_dir + self.time_id +
                "-input_columns.csv")

        self.data = self.df.to_numpy()

        self._split_train_test(test_equals_train=False)
        self._scale_data()

        # Save data for inspection of preprocessed data set
        self.df.to_csv("tmp_df.csv")
        np.savetxt("tmp_X_train.csv", self.X_train, delimiter=",")


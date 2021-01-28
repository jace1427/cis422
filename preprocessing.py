"""
Preprocessing funtions for Project 1 CIS 422. 

Authors: - Add your name if you help with something!
		 - Riley Matthews
		 - 

"""


# ------------------------ IMPORTS ----------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
import datetime
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})


# --------------------- PREPROCESSING -------------------- #

def denoise(ts: pd.DataFrame, window: int) -> pd.DataFrame:
	"""
	To denoise a timeseries we perform a rolling mean calculation.
	Window size should change depending on what the user wants to do with the data.

	//TODO: perhaps add different options for rolling calculations. Like use a rolling
	  median instead of the mean. 
	"""

	# the on= options makes the rolling() function use the first column of data
	# for the rolling window, rather than the DataFrame's index.
	ts_mean = ts.rolling(window, center=True, on=ts.columns[0]).mean()
	return ts_mean

def impute_missing_data(ts: pd.DataFrame, method: str) -> pd.DataFrame:
	"""
	This function fills in missing data in the timeseries.
	There are many ways to fill in blank data. We leave the choice to the user.
	options supported so far:
		- back fill: replace NaN with next known data
		- forward fill: replace NaN with last known data.
		- mean: replace NaN with the mean
		- median: replace NaN with the median
		- mode: replace NaN with the mode

	//TODO: Research ideas proposed by Kantz via project1 references.
	        TEST this method! Need to create datasets with missing values.
	"""
	if method == 'backfill':
		new_ts = ts.fillna(method='bfill')
	elif method == 'forwardfill':
		new_ts = ts.fillna(method='ffill')
	elif method == 'mean':
		new_ts = ts.fillna(ts.mean())
	elif method == 'median':
		new_ts = ts.fillna(ts.median())
	elif method == 'mode':
		new_ts = ts.fillna(ts.mnode())
	else:
		print("Method not supported")

	return new_ts

def impute_outliers(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Outliers are disparate data that we can treat as missing data. Use the same procedure
	as for missing data (sklearn implements outlier detection).

	//TODO: implement this function.
	"""

def longest_continuous_run(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Isolates the most extended portion of the time series without missing data.

	//TODO: implement this function.
	"""

def clip(ts: pd.DataFrame, starting_date: int, final_date: int) -> pd.DataFrame:
	"""
	This function clips the time series to the specified period's data.
	"""

	# The iloc pandas dataFrame method supports slicing.
	new_ts = ts.iloc[starting_date:final_date+1]
	return new_ts

def assign_time(ts, start, increment) -> pd.DataFrame:
	"""
	In many cases, we do not have the times associated with a sequence of readings.
	Start and increment represent to delta, respectively.

	//TODO: implement this function.
	"""

def difference(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Produces a time series whose magnitudes are the differenes between consecutive elements
	in the original time series.

	//TODO: implement this function.
	"""

def scaling(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Produces a time series whose magnitudes are scaled so that the resulting magnitudes range
	in the interval [0, 1].

	//TODO: implement this function.
	"""

def standardize(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Produces a time series whose mean is 0 and variance is 1.

	//TODO: implement this function.
	"""

def logarithm(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Produces a time series whose elements are the logarithm of the original elements.

	//TODO: implement this function.
	"""

def cubic_root(ts:pd.DataFrame) -> pd.DataFrame:
	"""
	Produces a time series whose elements are the original elements' cubic root.

	//TODO: implement this function.
	"""

def split_data(ts: pd.DataFrame, perc_training, perc_valid, perc_test):
	"""
	Splits a time series into training, validation, and testing according to the given
	percentages.

	//TODO: implement this function.
	        what is it supposed to return?
	        what are the types of perc_x?
	"""

def design_matrix(ts: pd.DataFrame, input_index, output_index):
	"""
	//TODO: find out what this is.
			implement it.
	"""

def design_matrix_2(ts, m0, m1, t0, t1):
	"""
	//TODO: find out what this is.
			implement it.
	"""

def ts2db(input_filename: str, perc_training, perc_valid, perc_test, input_index, output_index, output_file_name):
	"""
	The input index defines what part of the time series' history is designated to be the forcasting model's
	input. The forecasting task determines the output index, which indicates how many predictions are 
	required and how distanced they are from each other (not necessarily a constant distance).

	//TODO: input this function.
	"""
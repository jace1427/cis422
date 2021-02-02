"""
Preprocessing funtions for Project 1 CIS 422. 

Authors: - Add your name if you help with something!
		 - Riley Matthews
		 - Evan Podrabsky

"""


# ------------------------ IMPORTS ----------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import datetime
import math

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# Imports from Evan
import sklearn
import sklearn.model_selection
import csvreader

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
	        TEST this method! Need to create MORE datasets with missing values.
	"""

	# if there is no missing data. report and return original ts
	if ts.isnull().any().any() == False:
		print("impute_missing_data: ts has no missing data.")
		return ts

	# check with method the user requested
	if method == 'backfill':
		new_ts = ts.fillna(method='bfill')
	elif method == 'forwardfill':
		new_ts = ts.fillna(method='ffill')
	elif method == 'mean':
		new_ts = ts.fillna(ts.mean())
	elif method == 'median':
		new_ts = ts.fillna(ts.median())
	elif method == 'mode':
		new_ts = ts.fillna(ts.mode())
	else:
		print("Method not supported")

	return new_ts

def impute_outliers(ts: pd.DataFrame) -> pd.DataFrame:
	"""
	Outliers are disparate data that we can treat as missing data. Use the same procedure
	as for missing data (sklearn implements outlier detection).
	"""

	# generate list of z scores
	z_scores = stats.zscore(ts.iloc[:,1], nan_policy='omit')

	# first we must identify the outliers and mark them as NaN
	for i in range(len(ts.index)):
		if abs(z_scores[i]) > 2:
			ts.iloc[i, 1] = np.NaN

	# now we impute the missing data
	return impute_missing_data(ts, 'backfill')


def longest_continuous_run(ts: pd.DataFrame) -> pd.DataFrame:
	"""Isolates the most extended portion of the time series without missing data."""

	# the longest 
	start = 0
	end = 0
	longest = 0

	# the current
	start_curr = 0
	count = 0

	# visit each value and check if it is NaN or not
	for i in range(len(ts.index)):
		if pd.isna(ts.iloc[i, 1]):
			# we found a NaN value
			if count > longest:
				# This is a new longest run. Save it
				start = start_curr
				end = i - 1
				longest = count

				# reset current run
				start_curr = i + 1
				count = 0
			else:
				# we did not find a new longest run. reset current run
				start_curr = i + 1
				count = 0
		else:
			# found data
			count += 1
	
	# in the event the longest run of data is at the end of the dataFrame
	if count > longest:
		start = start_curr
		end = i
		longest = count

	# return the clip of the ts with the start and end values of the longest run
	return clip(ts, start, end)


def clip(ts: pd.DataFrame, starting_date: int, final_date: int) -> pd.DataFrame:
	"""
	This function clips the time series to the specified period's data.

	// TODO: the index numbers of the returned clip are the index numbers from the
			 original ts. Might introduce some bugs, test in future.
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
	"""Produces a time series whose elements are the logarithm of the original elements."""

	# visit each entry
	for i in range(len(ts.index)):
		# grab the value, compute ln(x), and store it back.
		x = ts.iloc[i, 1]
		ts.iloc[i, 1] = math.log(x)

	return ts

def cubic_root(ts:pd.DataFrame) -> pd.DataFrame:
	"""Produces a time series whose elements are the original elements' cubic root."""

	# visit each entry
	for i in range(len(ts.index)):
		# grab the value, compute x^(1/3), and store it back
		x = ts.iloc[i, 1]
		ts.iloc[i, 1] = x**(1./3.)

	return ts


def __split_data_usage(perc_training: float, perc_valid: float, perc_test: float):
	"""
	Usage test for split_data
	perc_training, perc_valid, and perc_test should add up to 1.0

	Parameters
	----------
	perc_training - float percentage in range [0.0, 1.0]
	perc_valid - float percentage in range [0.0, 1.0]
	perc_test - float percentage in range [0.0, 1.0]

	Returns
	-------
	None
	"""
	total_percentage = round(perc_training + perc_valid + perc_test, 3)
	if total_percentage != 1.0:
		print(f"ERROR: perc_training, perc_valid, and perc_test do not add up 1.0\n"
			  f"perc_training({perc_training})"
			  f" + perc_valid({perc_valid})"
			  f" + perc_test({perc_test}) = {total_percentage} != 1.0")
		raise ValueError


def split_data(ts: pd.DataFrame,
			   perc_training: float,
			   perc_valid: float,
			   perc_test: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
	"""
	Splits a time series into training, validation, and testing according to the given
	percentages.
	training_set, validation_set, and test_set are unique subsets of the main set (ts)
	which do not have any overlapping elements

	CALLS: clip()

	Parameters
	----------
	ts - Pandas.DataFrame object representing a time series (ts)

	perc_training - float percentage in range [0.0, 1.0]
	i.e. how many elements of the main set are divided into the training set

	perc_valid - float percentage in range [0.0, 1.0]
	i.e. how many elements of the main set are divided into the validation set

	perc_test - float percentage in range [0.0, 1.0]
	i.e. how many elements of the main set are divided into the test set


	Returns
	-------
	train_set, valid_set, test_set - tuple of three separate Pandas.DataFrame objects
	containing the three unique subsets of the main set
	"""

	# assert percents add up to 1.0
	__split_data_usage(perc_training, perc_valid, perc_test)

	num_cols = len(ts.columns)
	main_set_size = ts.size / num_cols

	# training set comprises bottom (perc_training) percentage of full set
	train_start = 0
	if main_set_size % 2 != 0:	# if we have an odd number of elements in the main set, add 1 extra to training set
		train_end = train_start + (round(perc_training * main_set_size))
	else:
		train_end = train_start + (round(perc_training * main_set_size) - 1)
	train_set = clip(ts, train_start, train_end)

	# validation set comprises first part of the remaining percentage where training set stopped
	valid_start = train_end + 1
	valid_end = valid_start + (round(perc_valid * main_set_size) - 1)
	valid_set = clip(ts, valid_start, valid_end)

	# test set comprises remaining percentage of the main set where validation set stopped
	test_start = valid_end + 1
	test_end = test_start + (round(perc_test * main_set_size) - 1)
	test_set = clip(ts, test_start, test_end)

	return train_set, valid_set, test_set


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

"""
Statistics and Visualization functions for Project 1 CIS 422.

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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import datetime
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})


# ------------- STATISTICS AND VISUALIZATION ------------- #

def myplot(*argv):
	"""
	Plots one of more time series.

	//TODO: Find out what "Adjust the time axis to display data according to their time indices."
	"""

	# there should only be one argument
	if len(argv) != 1:
		print('usage: myplot(ts | ts_list)')
		return None

	if type(argv[0]) is list:
		# loop through the list of ts and plot each one
		for frame in argv[0]:
			# use column[0] (time stamps) for x axis. column[1] (data) for y-axis.
			frame.plot(x=frame.columns[0], y=frame.columns[1])
		plt.show()
	else:
		ts = argv[0]
		ts.plot(x=ts.columns[0], y=ts.columns[1])
		plt.show()

def myhistogram(ts: pd.DataFrame):
	"""
	Compute and draw the histogram of the given time series.
	Plot the histogram vertically and side to side with a plot of the time series.
	"""
	ts.plot(x=ts.columns[0], y=ts.columns[1], kind='hist')
	plt.show()

def box_plot(ts: pd.DataFrame):
	"""
	Produces a Box and Whiskers plot of the time series. This function also prints the 
	5 number summary of the data.

	//TODO: The boxplot looks weird with some data. maybe use another method?
	"""
	# print 5 number summary
	print(ts.describe().iloc[3:8])

	# create box plot
	ts.plot(x=ts.columns[0], y=ts.columns[1], kind='box')
	plt.show()

def normality_test(ts: pd.DataFrame):
	"""
	Performs a hypothesis test about normality on the tine series data distribution.
	Besides the result of the statistical test, you may want to include a quantile plot
	of the data. Scipy contains the Shapiro-Wilkinson and other normality tests;
	matplotlib implements a qqplot function.

	//TODO: implement this function.
	        figure out what it should return.
	"""

def mse(y_test: pd.DataFrame, y_forecast: pd.DataFrame):
	"""
	Computes the MSE error of two time series.

	//TODO: implement this function.
	"""
	return ((y_forecast - y_test)**2).mean()

def mape(y_test: pd.DataFrame, y_forecast: pd.DataFrame):
	"""
	Computes the MAPE error of two time series

	//TODO: implement this function.
	"""
	return np.mean(np.abs((y_forecast - y_test) / y_test)) *100
def smape(y_test: pd.DataFrame, y_forecast: pd.DataFrame):
	"""
	Computes the SMAPE error of the two time series

	//TODO: implement this function.
	"""
	return 2.0 * np.mean(np.abs(y_forecast - y_true) / (np.abs(y_forecast) + np.abs(y_test))) * 100

"""
Preprocessing funtions for Project 1 CIS 422. 

Authors: - Add your name if you help with something!
         - Riley Matthews
         - Evan Podrabsky
         - Lannin Nakai

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
#mport csvreader

# Imports from Lannin
from sklearn.linear_model import LinearRegression

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})


# --------------------- PREPROCESSING -------------------- #

def denoise(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    To denoise a timeseries we perform a rolling mean calculation.
    Window size should change depending on what the user wants to do with the data.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.
    window: int
        an int indicating the window size of the rolling mean calculation.

    Returns
    -------
    ts_mean:
        a pandas DataFrame containing the modified time series data.
    """

    # the on= options makes the rolling() function use the first column of data
    # for the rolling window, rather than the DataFrame's index.
    ts_mean = ts.rolling(window, center=True, on=ts.columns[0]).mean()
    return ts_mean


def impute_missing_data(ts: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    This function fills in missing data in the timeseries.
    There are multiple ways to fill in blank data. We leave the choice to the user.
    options supported so far:
        - back fill: replace NaN with next known data
        - forward fill: replace NaN with last known data.
        - mean: replace NaN with the mean
        - median: replace NaN with the median
    
    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.
    method: a string
        indicates which method the user wants to use to fill in missing data.

    Returns
    -------
    new_ts: pd.DataFrame
        a pandas DataFrame containing the modified time series data.
    
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
        new_ts = ts.fillna(ts.mean(numeric_only=True))
    elif method == 'median':
        new_ts = ts.fillna(ts.median(numeric_only=True))
    else:
        print("Method not supported")

    return new_ts

def impute_outliers(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Outliers are disparate data that we can treat as missing data. Use the same procedure
    as for missing data (sklearn implements outlier detection).

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    a call to impute_missing_data (with a modified ts) which returns a pd.DataFrame
    -------
    """

    # generate list of z scores
    z_scores = stats.zscore(ts.iloc[:,1], nan_policy='omit')

    # first we must identify the outliers and mark them as NaN
    for i in range(len(ts.index)):
        if abs(z_scores[i]) > 2:
            ts.iloc[i, 1] = np.NaN

    if ts.isnull().any().any() == False:
        print("impute_outliers: ts has no outliers.")
        return ts

    # now we impute the missing data
    return impute_missing_data(ts, 'median')


def longest_continuous_run(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Isolates the most extended portion of the time series without missing data.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    a call to clip which returns a pd.DataFrame
    """

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

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.
    starting_date: int
        the starting index of the section to be clipped
    final_date: int
        the last index of the section to be clipped

    Returns
    -------
    new_ts: pd.DataFrame
        a pandas DataFrame that contains only the desired section of the original DataFrame
    """

    # check to see that the starting and final are in bounds.
    if (starting_date < 0) or starting_date>=len(ts.index):
        print('clip error: starting_date out of bounds.')

    if (final_date < 0) or final_date>=len(ts.index):
        print('clip error: final_date out of bounds.')

    # The iloc pandas dataFrame method supports slicing.
    new_ts = ts.iloc[starting_date:final_date+1]
    return new_ts


def assign_time(ts: pd.DataFrame, start: str, increment: int) -> pd.DataFrame:
    """
    In many cases, we do not have the times associated with a sequence of readings.
    Start and increment represent to delta, respectively.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.
    start: str
        a string that contains a starting date.
    increment: int
        an int indicating the number of hours between each timestamp.


    Returns
    -------
    ts: pd.DataFrame
        a pandas DataFrame containing the time series data with added timestamps.
    """

    # store the number of times we need to generate
    numtimes = len(ts.index)

    # create string to tell date_range how many hours in between each timestamp
    frequency = str(increment) + "H"

    # create a DatetimeIndex
    times = pd.date_range(start=start, periods=numtimes, freq=frequency)

    # insert our DatetimeIndex into the DataFrame as the first column
    ts.insert(0, 'Timestamp', times)

    return ts


def difference(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a time series whose magnitudes are the differenes between consecutive elements
    in the original time series.

    NOTE: if a list has 10 elements, there will be 9 differences. So the resulting list will 
    be shorter by 1. To avoid this we are proceding as follows:

    The difference between the ith element and the i-1 element is whats stored.

    When i=0 we just use the result from when i=1. 

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    new_ts: pd.DataFrame
        a pandas DataFrame containing the modified time series data.
    """

    # initialize empty list
    new_data = []

    # populate list with difference data
    for i in range(1, len(ts.index)):
        diff = ts.iloc[i, 1] - ts.iloc[i-1, 1]
        new_data.append(diff)

    # insert the first value. Same as second value, see docstring.
    new_data.insert(0, new_data[0])

    # give the Value column its new data values
    new_ts = ts.assign(Value=new_data)

    return new_ts


def scaling(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a time series whose magnitudes are scaled so that the resulting magnitudes range
    in the interval [0, 1].

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    ts: pd.DataFrame
        a pandas DataFrame containing the scaled time series data.
    """

    # create list of values
    values = ts.iloc[:, 1].to_list()

    # calculate the min and max values
    low = min(values)
    high = max(values)
    
    # visit each data entry and scale it
    for i in range(len(ts.index)):
        data = ts.iloc[i, 1]
        normalized = (data - low)/(high - low)
        ts.iloc[i, 1] = normalized

    return ts


def standardize(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a time series whose mean is 0 and variance is 1.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    ts: pd.DataFrame
        a pandas DataFrame containing the standardized time series data.
    """

    # find the mean
    mean = np.mean(ts['Value'])

    # find the standard deviation
    std = np.std(ts['Value'])

    # visit each data entry and standardize it
    for i in range(len(ts.index)):
        data = ts.iloc[i, 1]
        standardized = (data - mean)/(std)
        ts.iloc[i, 1] = standardized

    return ts

def logarithm(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a time series whose elements are the logarithm of the original elements.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    ts: pd.DataFrame
        a pandas DataFrame containing the modified time series data.
    """

    # visit each entry
    for i in range(len(ts.index)):
        # grab the value, compute ln(x), and store it back.
        x = ts.iloc[i, 1]
        ts.iloc[i, 1] = math.log(x)

    return ts

def cubic_root(ts:pd.DataFrame) -> pd.DataFrame:
    """
    Produces a time series whose elements are the original elements' cubic root.

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    ts: pd.DataFrame
        a pandas DataFrame containing the modified time series data.
    """

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
    if main_set_size % 2 != 0:  # if we have an odd number of elements in the main set, add 1 extra to training set
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


def design_matrix(ts: pd.DataFrame, input_index: list, output_index: list) -> list:
    """
        The input index defines what part of the time series' history is designated to be the 
        forcasting model's input. The forecasting task determines the output index, which indicates 
        how many predictions are required and how distanced they are from each other (not 
        necessarily a constant distance).

        Parameters
        ------------
        ts : pd.DataFrame
            dataframe we read from to get our design matrix values
        
        input_index : list
            list of positions in timeseries to be forecasting models input

        output_index : list
            list of positions in timeseries to be forecasting models desired output
    """
    
   
    input_array = []
    output_array = []
        
    t = max(input_array)
    stop_point = max(output_array) - stop_point

    for t in range(ts.index):

        for index in input_index:
            input_array.append(ts.data[t - index])

        for indec in output_index:
            output_array.append(ts.data[t + indec])
              
    return pd.DataFrame(data = [input_array, output_array])
    

def design_matrix_2(ts: list, mo: int, mi: int, to: int, ti: int):
    """
    See how well linear regressions determine the model fits

        Parameters
        -----------
        ts : pd.DataFrame
            pd.DataFrame holding our input and ouptut array
    
        mo : int
            matrix output value

        mi : int
            matrix input value

        to : int
            timestamp mo is taken at

        ti : int
            timestamp mi is taken at

        Return
        ----------
        score???
    ````"""    
         
    Xi = input_array[:, np.newaxis]
    Xo = output_array[:, np.newaxis]
    
        
    model = LinearRegression(fit_intercept=True)

    Xi = mi[:, np.newaxis]
    Xo = mo[:, np.newaxis]
      

  

def ts2db(input_filename: str, perc_training, perc_valid, perc_test, input_index, output_index, output_file_name):
    """
    Combines reading a file, splitting the data, converting to database, and producing the
        traiing databases.

    //TODO: input this function.
    """

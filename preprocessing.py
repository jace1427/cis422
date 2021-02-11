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
import fileIO

# Imports from Lannin
from sklearn.linear_model import LinearRegression

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})


# --------------------- PREPROCESSING -------------------- #

def denoise(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    To denoise a timeseries we perform a rolling mean calculation.
    Window size should change depending on what
    the user wants to do with the data.

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
    There are multiple ways to fill in blank data.
    We leave the choice to the user.
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
    Outliers are disparate data that we can treat as missing data.
    Use the same procedure as for missing data
    (sklearn implements outlier detection).

    Parameters
    ----------
    ts: pd.DataFrame
        a pandas DataFrame contaning time series data.

    Returns
    -------
    a call to impute_missing_data (with a modified ts)
    which returns a pd.DataFrame
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

    # return the clip of the ts with the
    # start and end values of the longest run
    return clip(ts, start, end)


def clip(ts: pd.DataFrame, starting_date: int,
         final_date: int) -> pd.DataFrame:
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
        a pandas DataFrame that contains only the
        desired section of the original DataFrame
    """

    # check to see that the starting and final are in bounds.
    if (starting_date < 0) or \
            starting_date>=len(ts.index) or \
            (starting_date > final_date):
        raise IndexError("clip - starting_date out of bounds")

    if final_date>=len(ts.index) or (final_date < starting_date):
        raise IndexError("clip - final_date out of bounds")

    # The iloc pandas dataFrame method supports slicing.
    new_ts = ts.iloc[starting_date:final_date+1]
    return new_ts


def assign_time(ts: pd.DataFrame, start: str, increment: int) -> pd.DataFrame:
    """
    In many cases, we do not have the times associated
    with a sequence of readings. Start and increment
    represent to delta, respectively.

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
        a pandas DataFrame containing the
        time series data with added timestamps.
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
    Produces a time series whose magnitudes are the
    differenes between consecutive elements in the original time series.

    NOTE: if a list has 10 elements, there will be 9 differences.
    So the resulting list will be shorter by 1.
    To avoid this we are proceding as follows:

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
    Produces a time series whose magnitudes are scaled so
    that the resulting magnitudes range
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
    Produces a time series whose elements are the
    logarithm of the original elements.

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
    Produces a time series whose elements are the
    original elements' cubic root.

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


def __split_data_usage(perc_training: float, perc_valid: float,
                       perc_test: float):
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
        print(f"ERROR: perc_training, perc_valid, "
              f"and perc_test do not add up 1.0\n"
              f"perc_training({perc_training})"
              f" + perc_valid({perc_valid})"
              f" + perc_test({perc_test}) = {total_percentage} != 1.0")
        raise ValueError


def split_data(ts: pd.DataFrame,
               perc_training: float,
               perc_valid: float,
               perc_test: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits a time series into training, validation,
    and testing according to the given percentages.
    training_set, validation_set, and test_set are
    unique subsets of the main set (ts) which do not
    have any overlapping elements

    CALLS: clip()

    Parameters
    ----------
    ts - Pandas.DataFrame object representing a time series (ts)

    perc_training : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set
        are divided into the training set

    perc_valid : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set
        are divided into the validation set

    perc_test : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set are divided into the test set


    Returns
    -------
    train_set, valid_set, test_set : tuple
        three separate Pandas.DataFrame objects
        containing the three unique subsets of the main set
    """

    # assert percents add up to 1.0
    __split_data_usage(perc_training, perc_valid, perc_test)

    num_cols = len(ts.columns)
    main_set_size = ts.size / num_cols

    # training set comprises bottom (perc_training) percentage of full set
    train_start = 0
    # if we have an odd number of elements in the main set,
    # add 1 extra to training set
    if main_set_size % 2 != 0:
        train_end = train_start + (round(perc_training * main_set_size))
    else:
        train_end = train_start + (round(perc_training * main_set_size) - 1)
    train_set = clip(ts, train_start, train_end)

    # validation set comprises first part
    # of the remaining percentage where training set stopped
    valid_start = train_end + 1
    valid_end = valid_start + (round(perc_valid * main_set_size) - 1)
    valid_set = clip(ts, valid_start, valid_end)

    # test set comprises remaining percentage
    # of the main set where validation set stopped
    test_start = valid_end + 1
    test_end = test_start + (round(perc_test * main_set_size) - 1)
    test_set = clip(ts, test_start, test_end)

    return train_set, valid_set, test_set


def design_matrix(ts: pd.DataFrame, input_index: list,
                  output_index: list) -> pd.DataFrame:
    """
        The input index defines what part of the time series'
        history is designated to be the forcasting model's input.
        The forecasting task determines the output index, which indicates
        how many predictions are required and how distanced they
        are from each other (not necessarily a constant distance).

        Parameters
        ------------
        ts : pd.DataFrame
            dataframe we read from to get our design matrix values

        input_index : list
            list of positions in timeseries to be forecasting models input

        output_index : list
            list of positions in timeseries to be
            forecasting models desired output

        Returns
        -----------
        design matrix : pandas.DataFrame
    """
    window = 8
    # can be modified according to preference
    
    # we allocate a dataframe for storing the design matrix
    df = pd.DataFrame(index=range(window), columns = input_index + output_index)

    steps = 3
    # can be modified according to preference
    
    
    # for the number of steps specified (hard-coded in steps)
    for i in range(steps):
        
        #create our training and prediction data
        for ind in input_index:
            train = ts.iloc[ind - window + i:ind + window + i]
        for ind in output_index:
            predict = ts.iloc[ind-window+i:ind + window + i]
        
        # variable to hold the compiled training and prediction values
        accum = [] 

        # attatch the training data
        for j in range(len(train)):
            accum.append(train.iloc[j][1])
        
        # attatch the prediction data
        for k in range(len(predict)):
            accum.append(predict.iloc[k][1])
        
        # place it all in a data frame
        for l in range(len(accum)):
            df.iloc[i,l] = accum[l]    
        
    return df


def design_matrix_2(ts: pd.DataFrame, mo: int, mi: int,
                    to: int, ti: int) -> pd.DataFrame:
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
    Design Matrix : pd.DataFrame
    """    
    
    # create a dataframe object with a size that will fit the design matrix
    df = pd.DataFrame(index=range(to-ti), columns =range(mi + mo))
    
    # for every step between the starting and ending time...
    for i in range(to-ti):
        
        # take a slice of data for both training and prediction
        # the size of the slices are determined by the sizes of
        # the input/output arrays (mi, mo)
        train = ts.iloc[ti + i:ti + i + mi]
        predict = ts.iloc[ti + i + mi:ti + i + mi + mo]

        # variable to hold the compiled training and prediction values
        accum = [] 

        # attatch the training data
        for j in range(len(train)):
            accum.append(train.iloc[j][1])
        
        # attatch the prediction data
        for k in range(len(predict)):
            accum.append(predict.iloc[k][1])
        
        # place it all in a data frame
        for l in range(len(accum)):
            df.at[i,l] = accum[l]    
    
    return df
    
    

def ts2db(input_filename: str, perc_training: float,
          perc_valid: float, perc_test: float, input_index: list,
          output_index: list,
          output_file_name: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Combines reading a file, splitting the data, converting to database,
    and producing the training databases.

    Parameters
    ----------
    intput_file_name : str
        csv file to read from

    perc_training : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set
        are divided into the training set

    perc_valid : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set
        are divided into the validation set

    perc_test : float
        percentage in range [0.0, 1.0]
        i.e. how many elements of the main set are divided into the test set

    input_index : list
        list of positions in timeseries to be forecasting models input

    output_index : list
        list of positions in timeseries to be
        forecasting models desired output

    output_file_name : str
        csv file to write training database to

    Return
    ------
    training_database : pandas.DataFrame,
        Training database to be used with estimator

    validation-set : pandas.DataFrame,
        Validation set to be used with estimator

    test_set : pandas.DataFrame
        Testing set to be used with estimator
    """

    # read csv file, split into train, valid, and test sets
    data = fileIO.read_from_file(input_filename)
    training_set, validation_set, test_set = split_data(data,
                                                        perc_training,
                                                        perc_valid,
                                                        perc_test)

    # get design matrix of training set based on input/output indices
    training_database = design_matrix(training_set,
                                      input_index,
                                      output_index)

    # pandas.DataFrame.to_csv will return None is successful
    status = training_database.to_csv(output_file_name)
    if status is not None:
        sys.stdout.write(f"{output_file_name} could not be found\n")
        raise FileNotFoundError
    else:
        sys.stdout.write(f"Training database written to file:"
                         f" {output_file_name}\n")

    # return tuple
    return training_database, validation_set, test_set

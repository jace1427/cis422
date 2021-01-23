"""
input and output for time series .csv files

Author: Evan Podrabsky

//TODO: Add module description
//TODO: Add csv format expectation doc for reporting error to users
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize': (11, 4)})


def csv_has_timestamps(file_name: str, sample_size: int) -> bool:
    """
    Checks the if the first column of a csv file appears
    to contain timestamps.

    Parameters
    ----------
    file_name - name of the csv file in the form of a string

    sample_size - how many lines in the csv to test (more is better)

    Returns
    -------
    bool - True if csv appears to have timestamps, False otherwise
    """

    # keep track of how many entries pandas
    # can and can't parse as datetime
    confirmed_timestamps = 0
    confirmed_non_timestamps = 0

    with open(file_name) as csvfile:
        for i in range(sample_size):
            try:

                # read each line from the file, prepare it,
                # then pass it to pandas.to_datetime()
                # if it can be parsed, we've encountered a timestamp entry
                line = csvfile.readline()
                line = line.rstrip()
                pd.to_datetime(line.split(",")[0])
                confirmed_timestamps += 1

            except ValueError:
                # pandas.to_datetime() will return ValueError if it can't
                # parse the argument, this means we've encountered a non-timestamp entry
                confirmed_non_timestamps += 1

    # very simple check: if we encountered more timestamps than not,
    # the primary entry for the first column must be timestamps
    if confirmed_timestamps > confirmed_non_timestamps:
        return True
    else:
        return False


def __csv_process_header(file_name: str) -> (int, list):
    """
    Finds how many lines of a csv file are
    taken up by the header(s).

    NOTE: csv file should be tested for timestamps
    with csv_has_timestamps()

    Parameters
    ----------
    file_name - name of the csv file in the form of a string

    Returns
    -------
    int - number of lines taken up by the header (0 if no header)
    list - list of the names of each column (will be modified if there are too many columns)
    """

    has_header = False
    header_lines = 0    # number of lines in the file before the data starts

    with open(file_name, "r") as csvfile:

        # set the top line string aside and focus on the top two lines
        top_line = csvfile.readline()
        previous_line = top_line
        current_line = csvfile.readline()

        # this loop will check if the first 4 characters of the string are the same
        # (which they should be if we're seeing a stream of data points w/ time stamps)
        while current_line:
            if previous_line[:3] == current_line[:3]:

                # if we've proven the existence of a header, we can now
                # utilize the saved top line in the names_list array we
                # pass to pandas.read_csv(), which will take the place
                # of the column headers
                if has_header:
                    top_line = top_line.rstrip()
                    names_list = top_line.split(',')

                # if no header exists, the current protocol is
                # to pass generic column names
                else:
                    names_list = ["Timestamp", "Value"]

                break   # break when we find where the data begins

            # if the latest two lines do not match, the file must have a header
            # move on to comparing the next pair of lines
            else:
                has_header = True
                previous_line = current_line
                current_line = csvfile.readline()
                header_lines += 1   # keep track of how many lines we go down

    return header_lines, names_list


def __csv_format_columns(file_name: str) -> None:
    """
    Creates a temporary copy of *file_name* as 'tmp.csv'
    which has all the same values as the original file,
    but the first two columns are merged.

    Parameters
    ----------
    file_name - name of the csv file in the form of a string

    Returns
    -------
    None

    Side Effects
    ------------
    A new file, "tmp.csv" is created in the working directory
    """
    # open original file and create new tmp file with write permissions
    original_file = open(file_name, "r")
    tmp_file = open("tmp.csv", "w")
    current_line = original_file.readline()

    # this loop will go through the original file,
    # copy each line and remove the first comma,
    # then write that line to the copy
    while current_line:

        # convert current_line to a list, remove the first comma,
        # convert list back into string, write to file
        line_list = list(current_line)
        comma_index = line_list.index(",")
        line_list[comma_index] = " "
        tmp_file.write("".join(line_list))

        # move to next line until we hit EOF
        current_line = original_file.readline()

    original_file.close()
    tmp_file.close()


def csv_to_dataframe(file_name: str) -> pd.DataFrame:
    """
    Reads csv file, checks if the file contains a header,
    and passes a file pointer at the start of the data to
    pandas.read_csv. This prevents any headers making
    it into the series values.

    Parameters
    ----------
    file_name - name of the csv file in the form of a string

    Returns
    -------
    pd.DataFrame - pandas DataFrame class holding data from csv file
    """

    formatted = False   # used to check if the columns needed to be formatted differently

    # check if csv contains timestamps,
    # find where the data starts if so, raise error otherwise
    if csv_has_timestamps(file_name, 50):
        data_start, names_list = __csv_process_header(file_name)

    else:
        print(f"{file_name} does not appear to have timestamps")
        raise TypeError

    # the number of entries in names_list corresponds to
    # the number of columns in the csv file
    num_columns = len(names_list)

    # more than three columns means this is not a time series
    if num_columns > 3:
        print(f"ERROR TOO MANY COLUMNS")
        raise NotImplementedError

    # exactly three columns is a special case where it is assumed the
    # first two columns are part of one timestamp
    elif num_columns == 3:

        # format the columns according to our specifications
        # then update variables as such
        __csv_format_columns(file_name)
        formatted = True
        file_name = "tmp.csv"

        # the first and second column will be merged
        names_list[0] = names_list[0] + " " + names_list[1]
        names_list.remove(names_list[1])

    # anything under 2 means this is not a time series with timestamps
    elif num_columns < 2:
        print(f"ERROR TOO FEW COLUMNS")
        raise NotImplementedError

    # open either the file from the 'file_name' argument
    # or 'tmp.csv' if column formatting was necessary
    file = open(file_name, "r")

    # based on return from csv_process_header(), move the
    # file pointer forward until we find the start of the data
    for i in range(data_start):
        file.readline()

    # finally, call the pandas library function
    data_frame = pd.read_csv(file,
                             header=None,
                             names=names_list,
                             index_col=0,
                             parse_dates=True)

    file.close()
    if formatted:
        os.remove(file_name)    # delete the tmp.csv file from the working directory
    return data_frame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

import matplotlib.dates as mdates

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA


def csv_to_dataframe(file_name: str) -> pd.DataFrame:
    """
    Reads csv file, checks if the file contains a header,
    and passes a file pointer at the start of the data to
    pandas.read_csv. This prevents any headers making
    it into the series values.

    Parameters
    ----------
    file_name

    Returns
    -------
    pd.DataFrame
    """

    has_header = False
    data_start = 0 # number of lines in the file before the data starts

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

                # if no header exists, the current protocol is to pass a
                # NULL array to pandas.read_csv()
                # this way, column headers in the returned DataFrame
                # will be labelled sequentially up from 0
                else:
                    names_list = None
                break # break when we find where the data begins

            # if the latest two lines do not match, the file must have a header
            # move on to comparing the next pair of lines
            else:
                has_header = True
                previous_line = current_line
                current_line = csvfile.readline()
                data_start += 1 # keep track of how many lines we go down

    # since passing around a file pointer in python is a ludicrous notion,
    # we need to re-open the file and, using the information we just found,
    # go through some lines until we find where the data starts,
    # then finally pass that pointer to pandas.read_csv()
    file = open(file_name, "r")
    for i in range(data_start):
        file.readline()
    data_frame = pd.read_csv(file,
                             header=None,
                             names=names_list,
                             index_col=0,
                             parse_dates=True)

    file.close()
    return data_frame

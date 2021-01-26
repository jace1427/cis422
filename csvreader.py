"""
input and output for time series .csv files

Author: Evan Podrabsky

THIS MODULE MAY BE REFACTORED IN THE FUTURE, BUT ITS
FUNCTIONALITY AND PUBLIC FUNCTIONS WILL REMAIN THE SAME

//TODO: Add module description
//TODO: Add csv format expectation doc for reporting error to users
"""

import pandas as pd
import numpy as np
import seaborn as sns
import datetime
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

    # check if csv contains timestamps,
    # find where the data starts if so, raise error otherwise
    if csv_has_timestamps(file_name, 50):
        data_start, names_list = __csv_process_header(file_name)
    else:
        print(f"{file_name} does not appear to have timestamps")
        raise TypeError

    file = open(file_name, "r")

    # based on return from __csv_process_header(), move the
    # file pointer forward until we find the start of the data
    for i in range(data_start):
        file.readline()

    # finally, call the pandas library function
    data_frame = pd.read_csv(file,
                             header=None,
                             names=names_list,
                             index_col=None,
                             parse_dates=True)

    # convert the relevant rows to timestamps and floats
    data_frame = __convert_timestamps(data_frame)

    file.close()
    return data_frame


def __convert_timestamps(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame object with 2-3 columns. Using a python builtin list
    to represent the DataFrame, this function converts all date/time-related columns
    to a single column of pandas.Timestamp objects. Columns related to time series
    values are converted to python builtin float objects.

    NOTE: This function is predicated on the assumption of
    dataframe having between 2 and 3 columns. It will throw a NotImplementedError
    if it finds more column labels than expected, but it does not check the contents
    of the DataFrame object.
    i.e. PERFORM YOUR ERROR CHECKING BEFORE CALLING THIS FUNCTION

    Parameters
    ----------
    data_frame - DataFrame object containing 2-3 columns of data
    (the first and second columns are assumed to be timestamp related while the third is assumed to be a float value)

    Returns
    -------
    pd.DataFrame - new DataFrame object with appropriate entry typing and column format
    """

    # get 2D array of data_frame as python builtin list, get columns list to use for second pd.DataFrame constructor
    raw_array = data_frame.to_numpy(dtype=str).tolist()
    columns_list = data_frame.columns.tolist()

    # num_rows used to iterate through array, num_cols used for naive error checking
    num_rows = len(raw_array)
    num_cols = len(raw_array[0])

    # here is why it is important to error check DataFrames for correct dimensions before passing to this function
    # __convert_two and convert_tree_cols() both have a different process and could hit errors
    # or produce undefined behavior
    if num_cols == 3:
        __convert_three_cols(raw_array, num_rows)
        columns_list.remove(columns_list[1])    # convert_three_cols deletes a column, update our list of column names
    elif num_cols == 2:
        __convert_two_cols(raw_array, num_rows)
    else:
        print(f"ERROR: GIVEN CSV FILE MUST CONTAIN TWO OR THREE COLUMNS (NOT A TIME SERIES)\n"
              f"(files with three columns are assumed to have date in the first column and a time in the second)")
        raise NotImplementedError

    # once columns have been processed, use array and list of column names
    # to reassign data_frame to a new constructor
    data_frame = pd.DataFrame(data=raw_array, columns=columns_list)

    return data_frame


def __convert_three_cols(raw_array: list, num_rows: int):
    """
    Converts the first and second column of a time series DataFrame into one column of timestamps
    (first two columns are assumed to be date/time-related).
    Also converts the last value-related column to floats.

    Parameters
    ----------
    raw_array - python builtin list representing a pandas.DataFrame object with exactly 3 columns
    num_rows - number of rows (entries) in the first dimension of the list

    Returns
    -------
    None
    """

    # check what we're working with in the second column and get it ready for us to join the two columns
    __process_times(raw_array, num_rows)

    # this loop does the actual conversions and gets rid of the second column altogether
    for i in range(num_rows):

        # float conversion
        float_value = float(raw_array[i][2])
        raw_array[i][2] = float_value

        # concatenate first two column entries, convert to timestamp, assign and delete redundancy
        timestamp = raw_array[i][0] + " " + raw_array[i][1]
        timestamp = pd.Timestamp(timestamp)
        raw_array[i].remove(raw_array[i][1])
        raw_array[i][0] = timestamp

    return


def __convert_two_cols(raw_array: list, num_rows: int):
    """
    Converts the first column of a time series DataFrame into a column of timestamps.
    Also converts the second value-related column to floats.

    Parameters
    ----------
    raw_array - python builtin list representing a pandas.DataFrame object with exactly 2 columns
    num_rows - number of rows (entries) in the first dimension of the list

    Returns
    -------
    None
    """

    # no need to check extra row, we can go right into conversions
    for i in range(num_rows):

        # float conversion
        float_value = float(raw_array[i][1])
        raw_array[i][1] = float_value

        # no need to delete an extra entry, we can just convert the existing string and assign it
        timestamp = pd.Timestamp(raw_array[i][0])
        raw_array[i][0] = timestamp

    return


def __process_times(raw_array: list, num_row: int):
    """
    Only called when there are 3 columns of data.
    This function checks if the second column of the DataFrame python list is comprised of integers,
    parsable pandas.Timestamp strings, or something not accepted as time-related. In either of the first
    two cases, the second column is reassigned to be parsable as the time of a pandas.Timestamp object.
    If neither of these cases are possible, TypeError is raised.

    Parameters
    ----------
    raw_array
    num_row

    Returns
    -------

    """

    # format strings used in datetime.time().strftime()
    full_time_format = '%H:%M:%S'
    hours_time_format = '%H:00:00'
    minutes_time_format = '00:%M:00'
    seconds_time_format = '00:00:%S'

    # booleans for telling what the integers in the second column represent
    integer_hours = False
    integer_minutes = False
    integer_seconds = False

    # boolean for if the second column is comprised of integers
    integer_timestamps = False

    # this loop breaks as soon as it finds an integer in the second column
    # if no integer is found, we try to parse it as a pandas.Timestamp,
    # if this fails, the entry is not something we can parse
    for i in range(num_row):
        time_string = raw_array[i][1]

        if time_string.isnumeric():
            integer_timestamps = True
            break

        try:
            time = pd.Timestamp(time_string)
            time = time.strftime(full_time_format)
            raw_array[i][1] = time

        except ValueError:
            print(f"ERROR: {time_string} cannot be parsed as a time value")
            raise TypeError

    # this will be the first thing we run into after encountering an integer in the previous loop
    if integer_timestamps:

        # we need to find what the maximum value is in the second column to decide if
        # it represents seconds, minutes, or hours
        max_value = 0
        min_value = 1

        # find the max and min value until the max value wraps around to 0
        # when this happens:
        # if max is 60, we're in minutes
        # if max is 24, we're in hours
        # else, we're in seconds
        for i in range(num_row):
            number = int(raw_array[i][1])

            if number < min_value:
                min_value = 0

            if number > max_value:
                max_value = number
            else:
                if number == min_value:
                    if (max_value == 60 and min_value == 1) or (max_value == 59 and min_value == 0):
                        integer_minutes = True
                    elif (max_value == 24 and min_value == 1) or (max_value == 23 and min_value == 0):
                        integer_hours = True
                    break

        # it's possible that we reached the end of the file without looping around to 0,
        # in this case, the max_value was never reached so we assume this column represents seconds
        # it is also possible that we wrapped around at 0 and just didn't meet the conditions
        # to consider the row as minutes or hours. We'll still use seconds in this case
        if max_value > 60:
            integer_seconds = True

        # basic switch case for putting each entry of second column into the desired format
        if integer_seconds:
            for i in range(num_row):
                number = int(raw_array[i][1])
                raw_array[i][1] = datetime.time(number).strftime(seconds_time_format)
        elif integer_minutes:
            for i in range(num_row):
                number = int(raw_array[i][1])
                raw_array[i][1] = datetime.time(number).strftime(minutes_time_format)
        elif integer_hours:
            for i in range(num_row):
                number = int(raw_array[i][1])
                raw_array[i][1] = datetime.time(number).strftime(hours_time_format)

    return


def read_from_file(file_name: str) -> pd.DataFrame:
    return csv_to_dataframe(file_name)

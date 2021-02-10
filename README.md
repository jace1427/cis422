# README

---
## INTRODUCTION

This library is designed for Data Scientists, the main functionallity is as follows:

* Multiple Pre-processing functions to mold data

* Multiple Statistics and visualization functions to be used on data

* CSV reading capabilities

* An mlp model

* A Tree Class, with functions passed into nodes

* A Pipeline Class, to be made from a tree that can execute the functions in the tree across data

---
## INSTALLATION AND USAGE

Install [Docker](https://www.docker.com/get-started)

Run this in a terminal: ```docker build -t <anyname> .```

This will automatically install any dependecies and create a new build.

The Dockerfile is currently pointed at "test.py", so you may either:
> Edit test.py with your code

> Change the Dockerfile to point at a .py file of your making

Executing the command: ```docker run <anyname>``` will run the contents of test.py.

To remove the Docker image, execute: ```docker rmi -f <anyname>```

### OR

install depedencies through pip or conda individually, create a file and import all from lib.py

---
## DOCUMENTATION

What follows is a brief desctription of each python file, and the functions or classes inside.

---
### estimator.py
Abstract Base Class that is used to inherit functions. Add methods to Override functions.

### fileIO.py
Functions:
> csv_has_timestamps(file_name: str, sample_size: int): Checks the if the first column of a csv file appears to contain timestamps. Refactored to include __csv_process_header functionality.

> csv_to_dataframe(file_name: str): Reads csv file, checks if the file contains a header, and passes a file pointer at the start of the data to pandas.read_csv. This prevents any headers making it into the series values.

> read_from_file(file_name: str): Wrapper for csv_to_dataframe.

> write_to_file(test_data: pd.DataFrame, estimator: Estimator, file_name: str): Write the predictions from the mlp model to the output file in the form of a csv with two columns (time, value).

### mlp_model.py
MLPModel: Basic multilayer perceptron model for time series. Inherits Estimator abstract class.
> fit(x_train: np.ndarray, y_train: np.ndarray): Train the MLP Regressor model to estimate values like those in y_train. Training is based off date values in x_train.

> score(x_valid: np.ndarray, y_valid: np.ndarray): Scores the model's prediction accuracy.

> forecast(x): Produces a forecast for the time series's current state.

> split_model_data(data: pd.DataFrame): Non-destructively splits a given data into sets readable for this estimator.

Functions:
> train_new_mlp_model(train_data: pd.DataFrame, input_dimension: int, output_dimension: int, layers=(5, 5)): Wrapper function for initializing and training a new mlp model.

> write_mlp_predictions_to_file(mlp: MLPModel, test_data: pd.DataFrame, file_name: str): Wrapper function for passing test data, estimator, and file name to fileIO.py.

### pipelines.py
Pipeline: Pipeline class for executing Trees
> save_pipeline(): Saves the current pipeline to the saved states history.

> make_pipeline(tree: Tree, route: list[int]): Creates a pipeline from a selected tree path.

> add_to_pipeline(func: function or list): Adds function(s) to pipeline, if it can connect.

> functions_connect(func_out : str, func_in : str): Checks that the preceding funcions output matching the proceeding functions input.

> run_pipeline(data : "database"): Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size).

### preprocessing.py
Functions:
> denoise(ts: pd.DataFrame, window: int): To denoise a timeseries we perform a rolling mean calculation. Window size should change depending on what the user wants to do with the data.

> impute_missing_data(ts: pd.DataFrame, method: str): This function fills in missing data in the timeseries. There are multiple ways to fill in blank data (see docstring).

> impute_outliers(ts: pd.DataFrame): Outliers are disparate data that we can treat as missing data. Use the same procedure as for missing data (sklearn implements outlier detection).

> longest_continuous_run(ts: pd.DataFrame): Isolates the most extended portion of the time series without missing data.

> clip(ts: pd.DataFrame, starting_date: int, final_date: int): This function clips the time series to the specified period's data.

> assign_time(ts: pd.DataFrame, start: str, increment: int): In many cases, we do not have the times associated with a sequence of readings. Start and increment represent to delta, respectively.

> difference(ts: pd.DataFrame): Produces a time series whose magnitudes are the differenes between consecutive elements in the original time series.

> scaling(ts: pd.DateFrame): Produces a time series whose magnitudes are scaled so that the resulting magnitudes range in the interval [0, 1].

> standardize(ts: pd.DataFrame): Produces a time series whose mean is 0 and variance is 1.

> logarithm(ts: pd.DataFrame): Produces a time series whose elements are the logarithm of the original elements.

> cubic_root(ts: pd.DataFrame): Produces a time series whose elements are the original elements' cubic root.

> split_data(ts: pd.DataFrame, perc_training: float, perc_valid: float, perc_test: float): Splits a time series into training, validation, and testing according to the given percentages. training_set, validation_set, and test_set are unique subsets of the main set (ts) which do not have any overlapping elements.

> desgin_matrix(ts: pd.DataFrame, input_index: list, output_index: list): The input index defines what part of the time series' history is designated to be the forcasting model's input. The forecasting task determines the output index, which indicates how many predictions are required and how distanced they are from each other (not necessarily a constant distance).

> design_matrix_2(ts: list, mo: int, mi: int, to: int, ti: int): See how well linear regressions determine the model fits.

> ts2db(input_filename: str, perc_training, perc_valid, perc_test, input_index, output_index, output_file_name): Combines reading a file, splitting the data, converting to database, and producing the traiing databases.

### statistics_and_visualization.py
Functions:
> myplot(*argv: list or tuple): Plots one of more time series.

> myhistogram(ts: pd.DataFrame): Compute and draw the histogram of the given time series. Plot the histogram vertically and side to side with a plot of the time series.

> box_plot(ts: pd.DataFrame): Produces a Box and Whiskers plot of the time series. This function also prints the 5 number summary of the data.

> normality_test(ts: pd.DataFrame): Performs a hypothesis test about normality on the tine series data distribution. Besides the result of the statistical test, you may want to include a quantile plot of the data. Scipy contains the Shapiro-Wilkinson and other normality tests; matplotlib implements a qqplot function. 

> mse(y_test: pd.DataFrame, y_forecast: pd.DataFrame): Computes the MSE error of two time series.

> mape(y_test: pd.DataFrame, y_forecast: pd.DataFrame): Computes the MAPE error of two time series

> smape(y_test: pd.DataFrame, y_forecast: pd.DataFrame): Computes the SMAPE error of the two time series

### tree.py
Node: A node class

Tree: N-ary tree
> search(target: int): Searches the tree for target, returns Node if found, otherwise None.

> insert(func: function, io_type: int, func_args: list[int], parent: int): Creates a new node with func to be inserted as a child of parent.

> delete(node: int): Deletes the given Node, children are removed.

> replace(node: int, func: function): Replaces the function of the given Node with func.

> get_args(parent: Node, child: Node): Gets function arguments the previous node does not fulfill.

> match(parent: Node, childe: Node): Checks if the child node can be attatched to the parent node matches child output to parent input.

> serialize(node: Node): Formats the tree into a string, so we can save the tree easily.

> deserialize(node: Node, saved_string: str): Returns the serialized string back to a tree.

> save_tree(): Saves the current tree state as a pre-ordered list.

> restore_tree(state_number: int): Saves the current tree state as a pre-ordered list.

> traverse(): Returns a preorder traversal of the Tree as a list.

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
## INSTALLATION

Install [Docker](https://www.docker.com/get-started)

Run this in a terminal: ```docker build -t <anyname> .```

This will automatically install any requirements and create a new build.

---
## TO USE

After installing requirements, simply import any python files you want to use.

---
## DOCUMENTATION

What follows is a brief desctription of each python file, and the functions or classes inside.

---
### estimator.py
Abstract Base Class that is used to inherit functions. Add methods to Override functions.

### fileIO.py
Functions:
> csv_has_timestamps: Checks the if the first column of a csv file appears to contain timestamps. Refactored to include __csv_process_header functionality.

> csv_to_dataframe: Reads csv file, checks if the file contains a header, and passes a file pointer at the start of the data to pandas.read_csv. This prevents any headers making it into the series values.

> read_from_file: Wrapper for csv_to_dataframe.

> write_to_file: Write the predictions from the mlp model to the output file in the form of a csv with two columns (time, value).

### mlp_model.py
MLPModel: Basic multilayer perceptron model for time series. Inherits Estimator abstract class.
> fit: Train the MLP Regressor model to estimate values like those in y_train. Training is based off date values in x_train.

> score: Scores the model's prediction accuracy.

> forecast: Produces a forecast for the time series's current state.

> split_model_data: Non-destructively splits a given data into sets readable for this estimator.

Functions:
> train_new_mlp_model: Wrapper function for initializing and training a new mlp model.

> write_mlp_predictions_to_file: Wrapper function for passing test data, estimator, and file name to fileIO.py.

### pipelines.py
Pipeline: Pipeline class for executing Trees
> save_pipeline: Saves the current pipeline to the saved states history.

> make_pipeline: Creates a pipeline from a selected tree path.

> add_to_pipeline: Adds function(s) to pipeline, if it can connect.

> functions_connect: Checks that the preceding funcions output matching the proceeding functions input.

> run_pipeline: Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size).

### preprocessing.py
Functions:
> denoise: To denoise a timeseries we perform a rolling mean calculation. Window size should change depending on what the user wants to do with the data.

> impute_missing_data: This function fills in missing data in the timeseries. There are multiple ways to fill in blank data (see docstring).

> impute_outliers: Outliers are disparate data that we can treat as missing data. Use the same procedure as for missing data (sklearn implements outlier detection).

> longest_continuous_run: Isolates the most extended portion of the time series without missing data.

> clip: This function clips the time series to the specified period's data.

> assign_time: In many cases, we do not have the times associated with a sequence of readings. Start and increment represent to delta, respectively.

> difference: Produces a time series whose magnitudes are the differenes between consecutive elements in the original time series.

> scaling: Produces a time series whose magnitudes are scaled so that the resulting magnitudes range in the interval [0, 1].

> standardize: Produces a time series whose mean is 0 and variance is 1.

> logarithm: Produces a time series whose elements are the logarithm of the original elements.

> cubic_root: Produces a time series whose elements are the original elements' cubic root.

> split_data: Splits a time series into training, validation, and testing according to the given percentages. training_set, validation_set, and test_set are unique subsets of the main set (ts) which do not have any overlapping elements.

> desgin_matrix: The input index defines what part of the time series' history is designated to be the forcasting model's input. The forecasting task determines the output index, which indicates how many predictions are required and how distanced they are from each other (not necessarily a constant distance).

> design_matrix_2: See how well linear regressions determine the model fits.

> ts2db: Combines reading a file, splitting the data, converting to database, and producing the traiing databases.

### statistics_and_visualization.py
Functions:
> myplot: Plots one of more time series.

> myhistogram: Compute and draw the histogram of the given time series. Plot the histogram vertically and side to side with a plot of the time series.

> box_plot: Produces a Box and Whiskers plot of the time series. This function also prints the 5 number summary of the data.

> normality_test: Performs a hypothesis test about normality on the tine series data distribution. Besides the result of the statistical test, you may want to include a quantile plot of the data. Scipy contains the Shapiro-Wilkinson and other normality tests; matplotlib implements a qqplot function. 

> mse: Computes the MSE error of two time series.

> mape: Computes the MAPE error of two time series

> smape: Computes the SMAPE error of the two time series

### tree.py
Node: A node class

Tree: N-ary tree
> search: Searches the tree for target, returns Node if found, otherwise None.

> insert: Creates a new node with func to be inserted as a child of parent.

> delete: Deletes the given Node, children are removed.

> replace: Replaces the function of the given Node with func.

> get_args: Gets function arguments the previous node does not fulfill.

> match: Checks if the child node can be attatched to the parent node matches child output to parent input.

> serialize: Formats the tree into a string, so we can save the tree easily.

> deserialize: Returns the serialized string back to a tree.

> save_tree: Saves the current tree state as a pre-ordered list.

> restore_tree: Saves the current tree state as a pre-ordered list.

> traverse: Returns a preorder traversal of the Tree as a list.

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
Run this in a terminal: 
```bash
docker build -t <anyname> .
```
This will automatically install any dependecies and create a new build.
The Dockerfile is currently pointed at "test.py", so you may either:
> Edit test.py with your code
> Change the Dockerfile to point at a .py file of your making
Executing the command will run the contents of test.py.
```bash
docker run <anyname>
``` 
will run the contents of test.py.
To remove the Docker image, execute: 
```bash
docker rmi -f <anyname>
```
#### OR
Install depedencies through pip or conda individually, create a file and import all from lib.py

---
## EXAMPLE
What follows is a Walkthrough detailing the creation a tree, adding several nodes, then creating and executing a pipeline.
We start by importing all from lib, which grants access to the library's functions and classes.
```python
from lib import *
```
\s\s
Then, Create a Tree.
```python
t = Tree()
```
\s\s
Begin inserting Nodes.
```python
t.insert(denoise, [1, 1], [5])
```
\s\s
The arguments for t.insert are as follows (See the docstring for more).
```python
func: 'function', io_type: int, func_args=[], parent=None
```
\s\s
Continue inserting.
```python
t.insert(impute_missing_data, [1, 1], func_args=["mean"], parent=0)
t.insert(logarithm, [1, 1], parent=1)
t.insert(split_data, [1, 2], func_args=[.25, .25, .5], parent=2)
```
\s\s
If you desire to use the MLP functionality, Create a new instance of the class MLPModel.
```python
mlp = MLPModel(1, 1)
```
\s\s
Then insert desired mlp functions.
```python
t.insert(train_new_mlp_model, [1, 3], func_args=[1, 1], parent=2)
t.insert(write_mlp_predictions_to_file, [3, 4], func_args=[data, "demo.csv"], parent=4)
```
\s\s
With a node to plot the data after running through the functions, the tree is finished.
```python
t.insert(myplot, [3, 5], parent=4)
```
\s\s
Next, Create a pipeline:
```python
p = Pipeline("exapmle")
```
\s\s
Make the Pipeline, specifying the pathway (See the docstring for more).
```python
p.make_pipeline(t, [0, 0, 1, 0])
```
\s\s
Create a Pandas DataFrame object out of your data. In this example, the data is stored in a csv, so we use csv_to_dataframe.
```python
data = csv_to_dataframe("TestData/test_data.csv")
```
\s\s
Run the Pipeline.
```python
p.run_pipeline(data)
```
\s\s
Execute the code using Docker.
```bash
docker build -t test .
```
\s\s
Run the code using Docker.
```bash
docker run test
```
\s\s
The output will be saved to a csv, ./demo.csv.To remove the docker image, execute this command.
```bash
docker rmi -f test
```

---
## DOCUMENTATION

What follows is a brief desctription of each python file, and the functions or classes inside.

### estimator.py
Abstract Base Class that is used to inherit functions. Add methods to Override functions.

### fileIO.py
Functions:
```python
csv_has_timestamps(file_name: str, sample_size: int)
```
Checks the if the first column of a csv file appears to contain timestamps. Refactored to include __csv_process_header functionality.\s\s

```python
csv_to_dataframe(file_name: str)
```
Reads csv file, checks if the file contains a header, and passes a file pointer at the start of the data to pandas.read_csv. This prevents any headers making it into the series values.\s\s

```python
read_from_file(file_name: str)
```
Wrapper for csv_to_dataframe.\s\s
```python
read_from_file_no_check(file_name: str)
```
Wrapper for pandas.read_csv(). Does not perform any checks of the data file.\s\s

```python
write_to_file(test_data: pd.DataFrame, estimator, Estimator, file_name: str)
```
Write the predictions from the mlp model to the output file in the form of a csv with two columns (time, value).\s\s

### mlp_model.py
MLPModel: Basic multilayer perceptron model for time series. Inherits Estimator abstract class.\s\s
```python
fit(x_train: np.ndarray, y_train: np.ndarray)
```
Train the MLP Regressor model to estimate values like those in y_train. Training is based off date values in x_train.\s\s

```python
score(x_valid: np.ndarray, y_valid: np.ndarray)
```
Scores the model's prediction accuracy.\s\s

```python
forecast(x)
```
Produces a forecast for the time series's current state.\s\s

```python
split_model_data(data: pd.DataFrame)
```
Non-destructively splits a given data into sets readable for this estimator.\s\s

Functions:
```python
train_new_mlp_model(train_data: pd.DataFrame, input_dimension: int, output_dimension: int, layers=(5, 5))
```
Wrapper function for initializing and training a new mlp model.\s\s

```python
write_mlp_predictions_to_file(mlp: MLPModel, test_data: pd.DataFrame, file_name: str)
```
Wrapper function for passing test data, estimator, and file name to fileIO.py.\s\s

### pipelines.py
Pipeline: Pipeline class for executing Trees.\s\s
```python
save_pipeline()
```
Saves the current pipeline to the saved states history.\s\s

```python
make_pipeline(tree: Tree, route: list[int])
```
Creates a pipeline from a selected tree path.\s\s

```python
add_to_pipeline(func: function or list)
```
Adds function(s) to pipeline, if it can connect.\s\s

```python
functions_connect(func_out : str, func_in : str)
```
Checks that the preceding funcions output matching the proceeding functions input.\s\s

```python
run_pipeline(data : "database")
```
Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size).\s\s

### preprocessing.py
Functions:
```python
denoise(ts: pd.DataFrame, window: int)
```
To denoise a timeseries we perform a rolling mean calculation. Window size should change depending on what the user wants to do with the data.\s\s

```python
impute_missing_data(ts: pd.DataFrame, method: str)
```
This function fills in missing data in the timeseries. There are multiple ways to fill in blank data (see docstring).\s\s

```python
impute_outliers(ts: pd.DataFrame)
```
Outliers are disparate data that we can treat as missing data. Use the same procedure as for missing data (sklearn implements outlier detection).\s\s

```python
longest_continuous_run(ts: pd.DataFrame)
```
Isolates the most extended portion of the time series without missing data.\s\s

```python
clip(ts: pd.DataFrame, starting_date: int, final_date: int)
```
This function clips the time series to the specified period's data.\s\s

```python
assign_time(ts: pd.DataFrame, start: str, increment: int)
```
In many cases, we do not have the times associated with a sequence of readings. Start and increment represent to delta, respectively.\s\s

```python
difference(ts: pd.DataFrame)
```
Produces a time series whose magnitudes are the differenes between consecutive elements in the original time series.\s\s

```python
scaling(ts: pd.DateFrame)
```
Produces a time series whose magnitudes are scaled so that the resulting magnitudes range in the interval [0, 1].\s\s

```python
standardize(ts: pd.DataFrame)
```
Produces a time series whose mean is 0 and variance is 1.\s\s

```python
logarithm(ts: pd.DataFrame)
```
Produces a time series whose elements are the logarithm of the original elements.\s\s

```python
cubic_root(ts: pd.DataFrame)
```
Produces a time series whose elements are the original elements' cubic root.\s\s

```python
split_data(ts: pd.DataFrame, perc_training: float, perc_valid: float, perc_test: float)
```
Splits a time series into training, validation, and testing according to the given percentages. training_set, validation_set, and test_set are unique subsets of the main set (ts) which do not have any overlapping elements.\s\s

```python
desgin_matrix(ts: pd.DataFrame, input_index: list, output_index: list)
```
The input index defines what part of the time series' history is designated to be the forcasting model's input. The forecasting task determines the output index, which indicates how many predictions are required and how distanced they are from each other (not necessarily a constant distance).\s\s

```python
design_matrix_2(ts: list, mo: int, mi: int, to: int, ti: int)
```
See how well linear regressions determine the model fits.\s\s

```python
ts2db(input_filename: str, perc_training, perc_valid, perc_test, input_index, output_index, output_file_name)
```
Combines reading a file, splitting the data, converting to database, and producing the traiing databases.\s\s

### statistics_and_visualization.py
Functions:
```python
myplot(*argv: list or tuple)
```
Plots one of more time series.\s\s

```python
myhistogram(ts: pd.DataFrame)
```
Compute and draw the histogram of the given time series. Plot the histogram vertically and side to side with a plot of the time series.\s\s

```python
box_plot(ts: pd.DataFrame)
```
Produces a Box and Whiskers plot of the time series. This function also prints the 5 number summary of the data.\s\s

```python
normality_test(ts: pd.DataFrame)
```
Performs a hypothesis test about normality on the tine series data distribution. Besides the result of the statistical test, you may want to include a quantile plot of the data. Scipy contains the Shapiro-Wilkinson and other normality tests; matplotlib implements a qqplot function.\s\s

```python
mse(y_test: pd.DataFrame, y_forecast: pd.DataFrame)
```
Computes the MSE error of two time series.\s\s

```python
mape(y_test: pd.DataFrame, y_forecast: pd.DataFrame)
```
Computes the MAPE error of two time series\s\s

```python
smape(y_test: pd.DataFrame, y_forecast: pd.DataFrame)
```
Computes the SMAPE error of the two time series\s\s

### tree.py
Node: A node class.

Tree: N-ary tree.
```python
search(target: int)
```
Searches the tree for target, returns Node if found, otherwise None.\s\s

```python
insert(func: function, io_type: int, func_args: list[int], parent: int)
```
Creates a new node with func to be inserted as a child of parent.\s\s

```python
delete(node: int)
```
Deletes the given Node, children are removed.\s\s

```python
replace(node: int, func: function)
```
Replaces the function of the given Node with func.\s\s

```python
get_args(parent: Node, child: Node)
```
Gets function arguments the previous node does not fulfill.\s\s

```python
match(parent: Node, childe: Node)
```
Checks if the child node can be attatched to the parent node matches child output to parent input.\s\s

```python
serialize(node: Node)
```
Formats the tree into a string, so we can save the tree easily.\s\s

```python
deserialize(node: Node, saved_string: str)
```
Returns the serialized string back to a tree.\s\s

```python
save_tree()
```
Saves the current tree state as a pre-ordered list.\s\s

```python
restore_tree(state_number: int)
```
Saves the current tree state as a pre-ordered list.\s\s

```python
traverse()
```
Returns a preorder traversal of the Tree as a list.\s\s

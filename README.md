# README

---

## INTRODUCTION

This library is designed for Data Scientists, the main functionallity is as follows:
- Multiple Pre-processing functions to mold data
- Multiple Statistics and visualization functions to be used on data
- CSV reading capabilities
- A Tree Class, with functions passed into nodes
- A Pipeline Class, to be made from a tree that can execute the functions in the tree across data

---

## INSTALLATION

Run this in a terminal: ```python3 Dockerfile``` (replace "python3" with "python" or "py" depending on your system). This will install all requirements for the library.

---

## TO USE

After installing requirements, simply import any python files you want to use.

---

## DOCUMENTATION

What follows is a brief desctription of each python file, and the functions or classes inside.

### estimator.py
Abstract Base Class that is used to inherit functions. Add methods to Override functions.

### fileIO.py

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

### preprocessing.py

### statistics_and_visualization.py

### tree.py
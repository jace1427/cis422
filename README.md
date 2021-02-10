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

Run this in a terminal: 

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
Pipeline: Pipeline class for executing Trees
> save_pipeline: Saves the current pipeline to the saved states history.

> make_pipeline: Creates a pipeline from a selected tree path.

> add_to_pipeline: Adds function(s) to pipeline, if it can connect.

> functions_connect: Checks that the preceding funcions output matching the proceeding functions input.

> run_pipeline: Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size).

### preprocessing.py

### statistics_and_visualization.py

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

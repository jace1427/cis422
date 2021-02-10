import numpy as np
import pandas as pd
import sklearn.neural_network as sknn
import copy
from estimator import Estimator
import fileIO


class MLPModel(Estimator):
    """
    Basic multilayer perceptron model for time series
    """

    def __init__(self, input_dimension: int, output_dimension: int, layers=(2, 2), *arguments):
        if not input_dimension or not output_dimension:
            print(f"ERROR (mlp_model.__init__()): input_dimension and output_dimension must both be non-zero")
            raise ValueError

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.layers = layers
        self.model = sknn.MLPRegressor(hidden_layer_sizes=layers,
                                       activation='relu',
                                       solver='adam',
                                       alpha=10,
                                       learning_rate='constant',
                                       max_iter=300,
                                       shuffle=True,
                                       random_state=1,
                                       momentum=0.9)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Train the MLP Regressor model to estimate values like those in y_train.
        Training is based off date values in x_train.

        Parameters
        ----------
        x_train {numpy.ndarray}: Training set containing date values to train with

        y_train {numpy.ndarray}: Training set containing actual float values for the times in x_train
        MLPRegressor uses these to make another pass over the data (another epoch) if some
        pattern convergence has not been found.

        Returns
        -------
        None
        """

        self.model.fit(x_train, y_train)

    def score(self, x_valid: np.ndarray, y_valid: np.ndarray) -> float:
        """
        Scores the model's prediction accuracy

        Parameters
        ----------
        x_valid {numpy.ndarray}: Validation set containing date values to guess values for

        y_valid {numpy.ndarray}: Validation set containing actual float values for the predicted times

        Returns
        -------
        score {float}: float value <= 1.0 representing accuracy of mlp_model predictions
        (can be negative if predictions are arbitrarily inaccurate)
        """

        return self.model.score(x_valid, y_valid)

    def forecast(self, x):
        """
        Produces a forecast for the time series's current state

        Parameters
        ----------
        current_state {numpy.ndarray OR one number}: (number can be numpy.int64 or numpy.float64)
        Current state to forecast for in the time series.
        If given as ndarray, returns ndarray of forecasts, else returns single forecast

        Returns
        -------
        forecast {numpy.ndarray of numpy.float64}:
        ndarray of forecasts for given states (may have just one element)
        """

        if isinstance(x, np.int64) or isinstance(x, np.float64):
            current_state = np.ndarray((1, 1), np.int64)
            current_state[0][0] = x
            prediction = self.model.predict(current_state)
            return prediction

        elif isinstance(x, np.ndarray) and len(x.shape) > 1:
            prediction = self.model.predict(x)
            return prediction

        else:
            print(f"INPUT ERROR: {repr(x)} is not an expected type\n"
                  f"mlp_model.forecast() expects a 2D numpy.ndarray of numpy.int64 elements or a single numpy.int64")
            raise TypeError

    def split_model_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Non-destructively splits a given data into sets readable for this estimator

        Parameters
        ----------
        set {pandas.DataFrame}: Dataframe with at least 2 columns of data

        Returns
        -------
        x_set, y_set {numpy.ndarray, numpy.ndarray}: x_set is the split of the original set containing
        the same number of columns as mlp's input dimension. y_set is the split of the original set
        which contains the same number of columns as mlp's output dimension.
        These numpy.ndarray objects can be passed directly to mlp.fit(x_set, y_set) or mlp.score(x_set, y_set)
        If mlp.output_dimension == 1, y_set will be flattened via numpy.ravel() into a 1D numpy.ndarray
        (this is necessary for the sklearn MLPRegressor model)
        """

        total_columns = len(data.columns)

        if self.input_dimension + self.output_dimension != total_columns:
            print(f"ERROR (MLPModel.split_model_data()): {repr(data)} "
                  f"does not fit the input/output dimensions of this mlp model "
                  f"({self.input_dimension}, {self.output_dimension})")
            raise ValueError

        data_copy = copy.deepcopy(data)

        x_set = data_copy.loc[:, data_copy.columns[:self.input_dimension]]

        y_set = data_copy.loc[:, data_copy.columns[self.input_dimension:]]

        x_set[x_set.columns[0]] = x_set[x_set.columns[0]].map(lambda x: x.toordinal())

        if self.output_dimension == 1:
            y_set = y_set.to_numpy().ravel()
        else:
            y_set = y_set.to_numpy()

        x_set = x_set.to_numpy()

        return x_set, y_set


def train_new_mlp_model(train_data: pd.DataFrame,
                   input_dimension: int,
                   output_dimension: int,
                   layers=(5, 5)) -> MLPModel:
    """
    Wrapper function for initializing and training a new mlp model

    Parameters
    ----------
    input_dimension {int}: number of variables in the x portion of the data
    output_dimension {int}: number of variables in the y portion of the data
    layers {(int,)}: number of nodes in each hidden layer (one hidden layer for each element in the tuple)

    Returns
    -------
    mlp {mlp_model}: trained multilayer perceptron object
    """

    mlp = MLPModel(input_dimension, output_dimension, layers)

    x_train, y_train = mlp.split_model_data(train_data)

    mlp.fit(x_train, y_train)

    return mlp


def write_mlp_predictions_to_file(mlp: MLPModel, test_data: pd.DataFrame, file_name: str):
    """
    Wrapper function for passing test data, estimator, and file name to fileIO.py

    Parameters
    ----------
    test_data {pandas.DataFrame}: testing split from the original data

    estimator {any model with .forecast method}: estimator to make predictions with (must be trained)

    output_file {str}: File name to write to.
    - will have .csv extension
    - will contain header based on test_data

    Returns
    -------
    None
    """
    fileIO.write_to_file(test_data, mlp, file_name)
    return

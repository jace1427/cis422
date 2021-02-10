class Estimator(object):
    """
    Abstract base class for any estimator to be used in a tree
    Implementation of new estimators must inherit this class.

    Methods
    -------
    fit
    forecast
    """

    def fit(self):
        """
        Trains the the estimator
        """
        pass

    def forecast(self):
        """
        Returns an estimation based on training data
        Estimator must always be trained first
        """
        return None

    def split_model_data(self):
        """
        Splits dataframe for estimator to train or predict from
        """
        return None

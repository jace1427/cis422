## MLP Model Test Suite
## MLP model output is unpredictable so this test suite
## does not include any tests for checking predictions

import unittest
import mlp_model
import fileIO
import preprocessing
import sys

# catches most testing from initialization to training to forecasting
# performance must be measured qualitatively
class Test_Create_New_MLP(unittest.TestCase):

    def test_raw_initialize(self):
        sys.stdout.write(f"Reading data...\n")
        data = fileIO.read_from_file("TestData/test_data.csv")
        training, validation, testing = preprocessing.split_data(data, 0.80, 0.10, 0.10)
        sys.stdout.write(f"Initializing new MLPModel object...\n")
        mlp = mlp_model.MLPModel(1, 1)
        x_train, y_train = mlp.split_model_data(training)
        sys.stdout.write(f"Training MLPModel...\n")
        mlp.fit(x_train, y_train)
        x_valid, y_valid = mlp.split_model_data(validation)
        x_test, y_test = mlp.split_model_data(testing)
        sys.stdout.write(f"Printing forecast from validation data...\n{mlp.forecast(x_valid)}\n\n"
                         f"Printing forecast from testing data...\n{mlp.forecast(x_test)}\n")

    def test_script_initialize(self):
        sys.stdout.write(f"Training new mlp model\n")
        data = fileIO.read_from_file("TestData/test_data.csv")
        training, validation, testing = preprocessing.split_data(data, 0.80, 0.10, 0.10)
        mlp = mlp_model.train_new_mlp_model(training, 1, 1)
        x_valid, y_valid = mlp.split_model_data(validation)
        x_test, y_test = mlp.split_model_data(testing)
        sys.stdout.write(f"Printing forecast from validation data...\n{mlp.forecast(x_valid)}\n\n"
                         f"Printing forecast from testing data...\n{mlp.forecast(x_test)}\n")


class Test_Split_Model_Data(unittest.TestCase):

    def test_wrong_input_dimension_right_output_dimension(self):
        data = fileIO.read_from_file("TestData/test_data.csv")
        training, validation, testing = preprocessing.split_data(data, 0.80, 0.10, 0.10)
        mlp = mlp_model.MLPModel(2, 1)
        with self.assertRaises(ValueError):
            x_train, y_train = mlp.split_model_data(training)

    def test_right_input_dimension_wrong_output_dimension(self):
        data = fileIO.read_from_file("TestData/test_data.csv")
        training, validation, testing = preprocessing.split_data(data, 0.80, 0.10, 0.10)
        mlp = mlp_model.MLPModel(1, 2)
        with self.assertRaises(ValueError):
            x_train, y_train = mlp.split_model_data(training)


if __name__ == "__main__":
    unittest.main()


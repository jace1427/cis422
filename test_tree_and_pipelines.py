# test tree and pipelines
from lib import *
import unittest

data = csv_to_dataframe("TestData/test_data.csv")
mlp = MLPModel(1, 1)


class T0_make_pipeline(unittest.TestCase):

    def test_make_pipeline(self):
        t = Tree()

        # preprocessing
        t.insert(denoise, [1, 1], [5])
        t.insert(impute_missing_data, [1, 1], func_args=["mean"], parent=0)
        t.insert(logarithm, [1, 1], parent=1)
        t.insert(split_data, [1, 2], func_args=[.25, .25, .5], parent=2)

        # mlp_model
        t.insert(train_new_mlp_model, [1, 3], func_args=[1, 1], parent=2)
        t.insert(write_mlp_predictions_to_file, [3, 4],
                 func_args=[data, "demo.csv"], parent=4)

        # statistics_and_visualization
        t.insert(myplot, [3, 5], parent=4)

        p = Pipeline("test")

        p.make_pipeline(t, [0, 0, 1, 0])
        p.run_pipeline(data)
        return


if __name__ == '__main__':
    unittest.main()

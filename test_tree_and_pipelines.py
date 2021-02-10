# test tree and pipelines
import unittest
from tree import *
from preprocessing import *
from mlp_model import *
from pipelines import *
from fileIO import *
from statistics_and_visualization import *

data = csv_to_dataframe("TestData/test_data.csv")
mlp = MLPModel(1, 1)


class T0_make_pipeline(unittest.TestCase):

    def test_make_pipeline(self):
        t = Tree()

        # pre-processing
        t.insert(denoise, [1, 1], [5])
        t.insert(impute_missing_data, [1, 1], ["mean"], 0)
        t.insert(logarithm, [1, 1], parent=1)
        t.insert(split_data, [1, 2], [.25, .25, .5], parent=2)

        t.insert(train_new_mlp_model, [1, 3], [1, 1], 2)

        t.insert(write_mlp_predictions_to_file, [3, 4], [data, "demo.csv"], 4)
        t.insert(myplot, [3, 5], parent=4)

        p = Pipeline("test")

        p.make_pipeline(t, [0, 0, 1, 0])
        p.run_pipeline(data)
        return


if __name__ == '__main__':
    unittest.main()
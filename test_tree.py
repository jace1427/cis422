# test tree
import unittest
from tree import *
from preprocessing import *
from mlp_model import *
from pipelines import *
from fileIO import *
from statistics_and_visualization import *

data = csv_to_dataframe("TestData/test_data.csv")

mlp = MLPModel(1,1)

"""
LEGEND
-------
1 = pd.DataFrame
2 = pd.DataFrame x 3
3 = MLPModel
4 = csv
5 = plot
...
..
.

Node:
    function
    [input,output]
    [function arguments not included in parent's return]
    parent (by node id #)
    
    Each node has unique args

"""
class T0_insert(unittest.TestCase):

    def test_insert(self):
        
        t = Tree()
        
        #pre-processing
        t.insert(denoise, [1,1], [5]) # Node 0
        t.insert(impute_missing_data, [1,1], ["mean"], 0) # Node 1
        t.insert(logarithm, [1,1], parent = 1) # Node 2
        t.insert(split_data, [1,2], [.25, .25, .5], parent = 2) # Node 3

        # branching starts
        # begin using mlp functionality
        t.insert(train_new_mlp_model, [1,3], [1,1], 2) # Node 4
        # branching starts
        
        t.insert(write_mlp_predictions_to_file, [3,4], [data, "demo.csv"], 4) # Node 5
        t.insert(myplot, [3,5], parent = 4) # Node 6

        p = Pipeline("test")
        
        p.make_pipeline(t, [0,0,1,0])
        p.run_pipeline(data)
        return

"""
class T1_delete(unittest.TestCase):
    def test_delete(self):
        t = Tree()
        t.insert(asdf)
        t.insert(qwer, 0)
        t.insert(zxcv, 1)
        t.insert(jkl, 2)
        t.insert(jkl, 1)
        t.insert(asdf, 0)
        t.insert(asdf, 4)
        t.delete(2)
        self.assertEqual(t.traverse(), [0, 1, 4, 6, 5])
        t.delete(4)
        self.assertEqual(t.traverse(), [0, 1, 5])


class T2_test_tree_with_funcs(unittest.TestCase):
    def test_tree_with_funcs(self):
        t = Tree()
        t.insert(denoise)
        t.insert(impute_missing_data, 0)
        t.insert(logarithm, 1)
        t.insert(split_model_data, 2)
"""

if __name__ == '__main__':
    unittest.main()

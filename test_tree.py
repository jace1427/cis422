# test tree

import unittest
from funcs import *
from tree import *


class T0_insert(unittest.TestCase):

    def test_insert(self):
        t = Tree()
        t.insert(asdf)
        t.insert(qwer, 0)
        t.insert(zxcv, 1)
        t.insert(jkl, 2)
        t.insert(jkl, 1)
        t.insert(asdf, 0)
        t.insert(asdf, 4)
        self.assertEqual(t.traverse(), [0, 1, 2, 3, 4, 6, 5])


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


if __name__ == '__main__':
    unittest.main()

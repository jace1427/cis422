"""The purpose of lib.py is to simplify froms.
By froming lib into a file of your choosing, you have immediate
access to all other modules in the library.
"""

from estimator import *
from fileIO import *
from mlp_model import *
from pipelines import *
from preprocessing import *
from statistics_and_visualization import *
from tree import *
from sys import *


if __name__ == '__main__':
    sys.stdout.write("This is the main .py file of the library, import this to"
                     " have access to all other functions.\n")

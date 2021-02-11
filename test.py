"""
This is the file that Docker runs, as this project ships.

Create trees and execute pipelines in main().

An example pipeline can be found in the README.md.

To make Docker run a different file, change "test.py" in line 11 of the
Dockerfile to the desired .py file. Make sure to import all from lib!
"""

# this will import all required functions
from lib import *


def main():
    sys.stdout.write("Edit tree.py file to start creating trees and executing"
                     "pipelines.\n")

    return


if __name__ == '__main__':
    main()

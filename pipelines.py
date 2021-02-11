######################
#       AUTHOR       #
######################
#    LANNIN NAKAI    #
######################

#####################
#    DESCRIPTION    #
####################################################
#  Handles functions used in pipelining to         #
#  organize/clean data to be used for forecasting  #
#  Program is ideally fed sequentially cohessive   #
#  data (as described in Wiley reading ch 12)      #
####################################################

from tree import *
from fileIO import *

# data = pd.read_csv() # get timeseries data
class Pipeline():
    """Pipeline class for executing Trees

    Attributes
    ----------
    name : str
        name of the pipeline

    Methods
    -------
    save_pipeline() -> None
    make_pipeline() -> None
    add_to_pipeline() -> None
    functions_connect() -> bool
    run_pipeline() -> database or bool
    """

    def __init__(self, name: str):
        # will be used as an identifier for the pipeline
        self.name = name

        # pipeline, a list of nodes that will be used to execute functions
        # sequentially
        self.pipeline = []

        # used for docmenting the history of the pipelines
        # composition/formation
        self.saved_states = []

    def save_pipeline(self) -> None:
        """Saves the current pipeline to the saved states history.

        Paramters
        ---------
        None

        Returns
        -------
        None
        """

        self.saved_states.append(self.pipeline)

    def make_pipeline(self, tree: Tree, route: [int]) -> None:
        """Creates a pipeline from a selected tree path.

        Parameters
        ----------
        tree : Tree
            tree we select our pipeline from

        route : [int]
            the route we take through the tree each number is the index of the
            parent node's children list which we want to travel down.

        Returns
        --------
        None
        """

        self.pipeline.append(tree.root)
        i = 0

        for entry in route:
            self.pipeline.append(self.pipeline[i].children[entry])
            i = i + 1

    def add_to_pipeline(self, func) -> None:
        """Adds function(s) to pipeline, if it can connect.

        Parameters
        ----------
        func : str
            function to be added to the pipeline

        Returns
        -------
        None
        """
        # Case 1: func is a list
        if len(func) > 1:

            for function in func:

                # before appending function, make sure it
                if functions_connect(self.pipeline[-1], function):

                    # will not have invalid input
                    # (disallowing it from the pipeline)
                    self.pipeline.append(function)

                else:
                    raise ValueError
                    sys.stdout.write(f"Sorry, but it seems that {function} "
                                     "does not fit into the pipeline. Check "
                                     f"{function} input and make sure it "
                                     f"matches {pipeline[-1]}'s output")

        # Case 2: func is not a list
        else:

            if functions_connect(self.pipeline[-1], function):

                self.pipeline.append(func)

            else:
                # message : printf("Sorry, but it seems that {} does not fit
                # into the pipeline. Check {} input and make sure it matches
                # {}'s output", function, function, pipeline[-1])
                raise ValueError

        return

    def functions_connect(self, func_out : str, func_in : str) -> bool:
        """Checks that the preceding funcions output matching the proceeding
        functions input.

        Parameters
        ----------
            func_out : str
                name of function data is being returned from
            func_in  : str
                function data is being stored to

        Returns
        -------
        True if functions match, else returns False
        """

        # search database for function matches
        if func_in in func_dict.funcs[func_out]:

            return True

        return False

    def run_pipeline(self, data : "database") -> "database" or bool:
        """Executes the functions stored in the pipeline sequentially
        (from 0 -> pipeline size)

        Parameters
        ----------
        data : database
            timeseries data to be processed

        Returns
        -------
        Data ran through pipeline, if successful, otherwise -1
        """
        # run the data through each function
        for node in self.pipeline:
            # case if there are additional args
            if node.func_args != []:
                # use output of the previous function as the input for the next
                inter_data = node.func(data, *node.func_args)
            else:
                inter_data = node.func(data)

            data = inter_data

        return data

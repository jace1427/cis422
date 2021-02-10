                ######################
                ##      AUTHOR      ## 
                ######################
                ##   LANNIN NAKAI   ##
                ######################

                #####################
                ##   DESCRIPTION   ##
####################################################
## Handles functions used in pipelining to        ##
## organize/clean data to be used for forecasting ##
## Program is ideally fed sequentially cohessive  ##
## data (as described in Wiley reading ch 12)     ##    
####################################################

from tree import *
from fileIO import *

# data = pd.read_csv() # get timeseries data
class Pipeline():

    def __init__(self, name: str):
        
        self.name = name
        # will be used as an identifier for the pipeline

        self.pipeline = []
        # pipeline, a list of nodes that will be used to execute functions sequentially
        
        self.saved_states = []
        # used for docmenting the history of the pipelines composition/formation

    def save_pipeline(self) -> None:
        # saves the current pipeline to the saved states history
        # Function Paramters
        #       - None
        # Returns
        #       - None

        self.saved_states.append(self.pipeline)

    def make_pipeline(self, tree: Tree, route: list) -> None:
        """ Creates a pipeline from a selected tree path

        Parameters
        ---------
        tree : Tree
            tree we select our pipeline from

        route : list of ints
            the route we take through the tree
            each number is the index of the parent
            node's children list which we want to 
            travel down. 

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
        # adds function(s) to pipeline if it can connect to the end of the pipeline
        #
        # Function Parameters
        #       - pipeline : list holding list of functions to be executed
        #       - func  : function to be added to the pipeline
        # Returns
        #       - None
        
        if len(func) > 1:
        # Case 1: func is a list
            
            for function in func:
            
                if functions_connect(self.pipeline[-1], function): 
                # before appending function, make sure it
                
                    self.pipeline.append(function) 
                    # will not have invalid input (disallowing it from the pipeline)
                
                else:
                    raise ValueError
                    printf("Sorry, but it seems that {} does not fit into the pipeline. Check {} input and make sure it matches {}'s output", function, function, pipeline[-1])
        else: 
        # Case 2: func is not a list
            
            if functions_connect(self.pipeline[-1], function):
            
                self.pipeline.append(func)
            
            else:

                raise ValueError
                # message : printf("Sorry, but it seems that {} does not fit into the pipeline. Check {} input and make sure it matches {}'s output", function, function, pipeline[-1])
        
        return
     

    def functions_connect(self, func_out : str, func_in : str) -> bool:
        # Checks that the preceding funcions output matching the proceeding functions input
        # The check is done by referencing a database of what functions 
        #
        # Function Parameters
        #       - func_out : name of function data is being returned from
        #       - func_in  : function data is being stored to
        # Returns
        #       - True if functions match, else returns False
        
        if func_in in func_dict.funcs[func_out]:
        # search database for function matches
            
            return True
        
        return False

    def run_pipeline(self, data : "database") -> "database" or bool:
        # Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size)
        #
        # Function Parameters
        #       - data : timeseries data to be processed
        # Returns
        #       - Data ran through pipeline, if successful
        #       - -1 , if unsuccessful


        for node in self.pipeline: 
        # run the data through each function
            if node.func_args != []:
            # case if there are additional args
                inter_data = node.func(data, *node.func_args) 
                # use the output of the previous function as the input for the next
            else:
                inter_data = node.func(data)

            data = inter_data
        
        return data


           































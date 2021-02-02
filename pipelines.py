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

    ###########
    ## TO DO ##
###################################################
## - MAKE PIPELINE CLASS INSTEAD OF LIST (MAYBE) ##
##                                               ##
## - ADD FUNCTIONALITY TO functions_connenct     ##
###################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
# from statsmodels.graphics.tsaplots import plot_pcaf, plot_acf
from statsmodels.tsa.arima.model import ARIMA


# data = pd.read_csv() # get timeseries data
class Pipeline():

    def __init__(self, name: str):
        
        self.name = name
        # will be used as an identifier for the pipeline

        self.pipeline = []
        # pipeline, a list that will execute functions sequentially
        
        self.saved_states = []
        # used for docmenting the history of the pipelines composition/formation

    def save_pipeline(self) -> None:
        # saves the current pipeline to the saved states history
        # Function Paramters
        #       - None
        # Returns
        #       - None

        self.saved_states.append(self.pipeline)

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


        for function in self.pipeline: 
        # run the data through each function
            
            inter_data = function(data) 
            # use the output of the previous function as the input for the next
            
            data = inter_data
        
        return data


           































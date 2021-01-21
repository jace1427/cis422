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
from statsmodels.graphics.tsaplots import plot_pcaf, plot_acf
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv() # get timeseries data


def make_pipeline():
    # makes a pipeline, a list that will execute functions sequentially
    #
    # Function Parameters
    #   - None
    # Returns
    #   - Empty list

    pipeline = []
    return pipeline

def add_to_pipeline(pipeline, func):
    # adds function(s) to pipeline if it can connect to the end of the pipeline
    #
    # Function Parameters
    #       - pipeline : list holding list of functions to be executed
    #       - func  : function to be added to the pipeline
    # Returns
    #       - None
    
    if len(func) > 1: # Case 1: func is a list
        for function in func:
            if functions_connect(pipeline[-1], function): # before appending function, make sure it
                pipeline.append(function) # will not have invalid input (disallowing it from the pipeline)
            else:
                printf("Sorry, but it seems that {} does not fit into the pipeline. Check {} input and make sure it matches {}'s output", function, function, pipeline[-1])
    else: # Case 2: func is not a list
        if functions_connect(pipeline[-1], function):
            pipeline.append(func)
        else:
            printf("Sorry, but it seems that {} does not fit into the pipeline. Check {} input and make sure it matches {}'s output", function, function, pipeline[-1])
    return
 

def functions_connect(func_out, func_in):
    # Checks to make sure that the preceding funcions output matching the proceeding functions input
    #
    # Function Parameters
    #       - func_out : function data is being returned from
    #       - func_in  : function data is being stored to
    # Returns
    #       - True if functions match, else returns False
    
    if func_out.argc == func_in.argc:
        return True
    return False

def run_pipeline(data, pipeline):
    # Executes the functions stored in the pipeline sequentially (from 0 -> pipeline size)
    #
    # Function Parameters
    #       - pipeline : list of functions
    # Returns
    #       - Data ran through pipeline, if successful
    #       - -1 , if unsuccessful


    for function in pipeline: # run the data through each function
        inter_data = function(data) # use the output of the previous function as the input for the next
        data = inter_data
    return data


       































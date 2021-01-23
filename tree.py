##################
##    AUTHOR    ##
##################
##################
## LANNIN NAKAI ##
##################

                    #################
                    ## DESCRIPTION ##
############################################################
## This script creates the transformation tree from which ##
## the user can customize the features of their pipeline  ##
############################################################

    ###########
    ## TO DO ##
    ###########
##################################################
## - ADD TREE_STATES[] ARRAY IN WHICH TREES     ##
## CAN BE SAVED (MAYBE DO IT IN DIFFERENT FILE) ##
##                                              ##
## - ADD CONNECTIVITY TO PIPELINE               ##
##################################################


####################
##   TREE CLASS   ##
####################

class Tree():
    # Tree (equivalent to multiple connected pipelines) class, which will hold nodes (functions of the pipeline)
    
    def __init__(self):
        self.root = None
        self.edit = 0

    def change_parent(self, children, node):
        # changes the parent of a list of children
        # Function Parameters
        #   - children : list of children
        #   - node : node to be named parent of children
        for child in children:
            child.parent = node
        return

    
    def insert_at(self, target, node, child = None):
        # target is the node that the node named "node" wil
        # Function Parameters:
        #   - target : node to be the parent of the new child
        #   - node : a new child to be attached the tree
        #   - child : node that will become the child of node
        # Return
        #   - None

        node.parent = target # set node as child of target

        if isEmpty(target.children): # Case 1 : target has no children
            target.children.append(node) #add the node to the target's childrens list
            return

        if child == None:  # Case 2 : child not specified
            node.children = target.children # set the nodes children to the target's children
            change_parent(node.children, node) # change the parent of target's children to node
            target.children = [node] # make node target's only child
            return
        
        # Case 3 : child specified
        node.children = [child] # set the nodes children to the child
        target.children.pop(child) # remove child from targets children
        child.parent = node # set the childs parent to node
        return
 
    def replace(self, target, node):
        # target is the node that the node named "node" will replace in the tree
        # Function Parameters:
        #   - target : node to be replace
        #   - node : a new node to be attached the tree
        # Return
        #   - None
        
        # ALTERNATIVELY, TRY TESTING WITH JUST THIS LINE FOR THE FUNCTION: target.func = node.func
        
        if !search(target):
            printf("target node {} does not exist \n", target)
            return

        node.parent = target.parent # replace parents of node with parents of target
       
        target.parent.children.append(node) # add node to target's parent's childrens list
        target.parent.children.pop(target) # remove target from its parent's children list
        if isEmpty(target.children):
            return

        node.children = target.children # let node take target's children
        change_parent(node.children, node) # set node the the target's childrens' parent
        return

    def search(self, func, start = self.root):
        # search the tree for a specified node
        # Function Parameters:
        #   - func : the function we are searching for
        #   - start : node to be searched from in the tree
        # Return
        #   - Boolean : True if found, else False

        if start.func != func: # Case 1 : node is not at start
            if !isEmpty(start.children): # Case 2 : There are still children to check for node in
                for child in start.children:
                    result = self.search(func, child) #recursively search through all branches
                    if result != False: # Case 3 : Node has been found
                        return True
        else: # Case 4 : Node is at start
            return start # if start.func == func
        
        return False # Case 5 : failure to find node


####################
##   NODE CLASS   ##
####################

class Node():
    # Node class will hold the functions as well as its relative position in relation to pipelines

    def __init__(self, func):
        self.func = func # func is an identifier and useful data for pipeline execution
        self.parent = None
        self.children = [] # children is held as a list since each node can have multiple children in an n-ary tree





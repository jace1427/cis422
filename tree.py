##################
#    AUTHORS     #
##################
##################
#  LANNIN NAKAI  #
# JUSTIN SPIDELL #
##################

#################
#  DESCRIPTION  #
############################################################
#  This script creates the transformation tree from which  #
#  the user can customize the features of their pipeline   #
############################################################

#############
#  IMPORTS  #
#############

import sys
# Allows script to access functions to place into the nodes

####################
#    NODE CLASS    #
####################


class Node():
    """A Node

    ...

    Attributes
    ----------

    num : int
        Number identification

    func : function
        The Node's function

    input_output_type: list (of two ints)
        Determines what type of inputs/outputs the function takes/returns

    func_args : list
        List of the arguments that wil lbe plugged into this node's function
        Necessary since some functions have more parameters than their parent
        node will be able to provide. This allows the user to specify those
        arguments while initializing the node

    parent : Node
        Pointer to the parent Node

    children : list[Node]
        List of pointers to children Nodes

    Methods
    -------
    None
    """

    def __init__(self, num, func, input_output_type, func_args = [], parent=None):
        self.num = num
        self.func = func
        self.parent = parent
        self.io_type = input_output_type
        self.func_args = func_args
        self.children = []

    def __repr__(self):
        return f"{self.num}: {self.children}"


####################
#    TREE CLASS    #
####################


class Tree():
    """N-ary tree

    ...

    Attributes
    ----------

    root : Node
        A pointer to the root Node of the Tree

    edit: Bool
        Bool repersenting whether or not the tree can be edited

    ctr: int
        Abritrary number for assigning id's to the Nodes

    string : str
        holder for serialization methods (serialize and deserialize)

    Methods
    -------
    search(target: int) -> Node or None:
        Searches the tree for target, returns Node if found, otherwise None.

    insert(func: function, parent: int) -> None:
        Creates a new node with func to be inserted as a child of parent.

    delete(node: int) -> None:
        Deletes the given Node, children are removed.

    replace(node: Node, func: function) -> None:
        Replaces the function of the given Node with func.

    traverse() -> list:
        Returns a preorder traversal of the Tree as a list.

    match() -> bool
        Checks if the child node can be attatched to the parent node matches
        child output to parent input.
    """

    def __init__(self):
        self.root = None
        self.edit = True
        self.saved_states = []
        self.ctr = 0
        self.string = ""

    def __repr__(self):
        return self.root.__repr__()

    def _searcher(self, target: int, curr: Node) -> None or Node:
        """Search helper method"""
        if curr.num == target:
            return curr

        for child in curr.children:
            tmp = self._searcher(target, child)
            if tmp is not None:
                return tmp

        return None

    def search(self, target: int) -> None or Node:
        """Searches the tree for target, returns Node if found, otherwise None.

        Parameters
        ----------
        target : int
            The number id of the node to be found

        Returns
        -------
        Description of the return value
        """
        curr = self.root

        return self._searcher(target, curr)

    def insert(self, func: 'function', io_type: int, func_args=[], parent=None) -> None:
        """Creates a new node with function func to be inserted
        as a child of parent.

        If no arg is given for parent, node will be made the root
        of the tree

        PLEASE: only attatch nodes to leaves

        Parameters
        ----------

        func : function
            function for new node to have

        parent : int
            parent of new Node, if None Node is inserted as the root

        Returns
        -------
        None
        """
        
        # check if a parent was given
        if parent is not None:
            
            # find the parent node in the tree
            parent_node = self.search(parent)

            # in the case an invalid parent is given, return an error
            if parent_node is None:
                sys.stderr.write(f"[ERROR]: Node ({parent}) not found\n")
                raise ValueError
                return

            # create a node with the information provided in function args
            node = Node(self.ctr, func, io_type, func_args, parent_node)
            
            # if the node does not fit the parent function, return an error
            if not self.match(parent_node, node):
                sys.stdout.write("Node cannot be attatched to targeted parent")
                sys.stdout.write(" invalid matching of args)\n")
                return

            # increment the tree's counter for node IDs
            self.ctr += 1

            # add the node in its new parent's list of children
            parent_node.children.append(node)

        # If parent is not specified, we make it the root
        else:
            
            # create a node with function args
            node = Node(self.ctr, func, io_type, func_args)
            
            # increment node ID counter
            self.ctr += 1
            
            if self.root is not None:
                # Make our node the root's childrens' new parent
                
                for child in self.root.children:
                    
                    child.parent = node
            self.root = node
        
        return

    def match(self, parent: Node, child: Node) -> bool:
        """Checks if the child node can be attatched to the parent node
        matches child output to parent input.

        Parameters
        -----------

        parent : Node
            Node we want to attach child node to

        child : Node
            Node we want to attatch to parent

        Returns
        ---------
        bool
        """
        # check that the childs output matches the functions input
        if parent.io_type[1] == child.io_type[0]:
            return True

        return False
"""
    DEPRECIATED FUNCTION (DUE TO IMPLEMENTING CODING SYSTEM)
    THAT MAY HAVE USE LATER ON
    
    def get_args(self, parent: Node, child: Node) -> list:
        Gets function arguments the previous node does not fulfill.

        Parameters
        -----------

        parent : Node
            Node we want to attach child node to

        child : Node
            Node we want to attatch to parent

        Returns
        ---------
        list : holding the arguments for our function
        
        # List of arguments to be added to the child function initialization
        args = []

        for item in inspect.signature(child.func).annotation:
            # if the argument isn't filled out by parent's return parameter(s)
            if item not in inspect.signature(parent.func).return_annotation:
                arg = input(f"Please enter argument for {item} : ")
                args.append(item)

        return args
"""

    def delete(self, node: int) -> None:
        """Deletes the given Node, children are removed.

        Parameters
        ----------
        node : int
            Node to be deleted

        Returns
        -------
        None
        """
        node = self.search(node)
        if node is None:
            raise ValueError
            return
        node.parent.children.remove(node)
        return

    def replace(self, node: int, func: 'function') -> None:
        """Replaces the function of the given Node with func.

        Parameters
        ----------
        node : int
            Node to be edited

        func : function
            function to be inserted

        Returns
        -------
        None
        """

        node = self.search(node)
        node.func = func
        return

    def serialize(self, node) -> None:
        """Formats the tree into a string, so we can save the tree easily.

        Parameters
        ----------
        node : Node
            iterates through nodes of tree

        Returns
        -------
        None
        """
        if node is None:
            return

        self.string.append([self.root.func, self.root.num])

        for child in node.children:
            self.serialize(child, self.string)

        self.string = self.string + ')'

        return

    def deserialize(self, node, saved_string) -> int:
        """Returns the serialized string back to a tree.

        Parameters
        ----------
        node : Node
            iterates through nodes of tree

        saved_string : str
            serialized string to be turned into a tree

        Returns
        -------
        int : 1 if success , 0 for failure
        """
        string = saved_string[0]
        saved_string = saved_string[1:]
        # basically popping the string

        if saved_string[0] == ')':
            return 1

        string.append(saved_string[0])
        self.string = self.string[1:]

        # references dictionary of functions ( keyed with letters)
        node = Node(string[0][0], func_list[string[0][1])

        for child in node.children:

            if self.deserialize(child, saved_string):
                break

        return 0

    def save_tree(self) -> None:
        """Saves the current tree state as a pre-ordered list.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # save the serialized string
        self.serialize(self.root)

        # reset the trees string for the next save
        self.saved_states.append(self.string)

        self.string = ""

        return

    def restore_tree(self, state_number) -> None:
        """Saves the current tree state as a pre-ordered list.

        Parameters
        ----------
        state_number : int
            version of saved tree that we will restore

        Returns
        ----------
        None
        """
        # the tree is saved here from the root
        self.deserialize(self.root, self.saved_state[state_number])

        return

    def _preorder(self, curr: Node, result: list) -> list:
        """Stores the tree as a list of nodes in preorder.

        Parameters
        ----------
        curr : Node
            current node being traversed

        result : list
            list we store the preordering on

        Returns
        -----------
        result : list
            list we store the preordering on
        """
        if curr:
            result.append(curr.num)
            for child in curr.children:
                self._preorder(child, result)

    def traverse(self) -> list:
        """Returns a preorder traversal of the Tree as a list.

        Parameters
        ----------
        None

        Returns
        -------
        Returns a list of ints
        """
        result = []
        self._preorder(self.root, result)
        return result

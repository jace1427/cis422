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

###########
#  TO DO  #
###########
##################################################
#  - ADD TREE_STATES[] ARRAY IN WHICH TREES      #
#  CAN BE SAVED (MAYBE DO IT IN DIFFERENT FILE)  #
#                                                #
#  - ADD CONNECTIVITY TO PIPELINE                #
##################################################
import sys


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
    
    parent : Node
        Pointer to the parent Node
    
    children : list[Node]
        List of pointers to children Nodes

    Methods
    -------
    None
    """

    def __init__(self, num, func, parent=None):
        self.num = num
        self.func = func
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"{self.num}: {self.children}"


####################
#    TREE CLASS    #
####################


class Tree():
    """Binary Search Tree.

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
        Searches the tree for target, returns Node if found, otherwise None

    insert(func: function, parent: int) -> None:
        Creates a new node with func to be inserted as a child of parent
    
    delete(node: int) -> None:
        Deletes the given Node, children are removed
    
    replace(node: Node, func: function) -> None:
        Replaces the function of the given Node with func
    
    traverse() -> list:
        Returns a preorder traversal of the Tree as a list
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
        """Searches the tree for target, returns Node if found, otherwise None

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

    def insert(self, func: 'function', parent=None) -> None:
        """Creates a new node with function func to be inserted
        as a child of parent

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
        if parent is not None:
            parent_node = self.search(parent)
            if parent_node is None:
                sys.stderr.write(f"[ERROR]: Node ({parent}) not found\n")
                raise ValueError # or some kind of error here to signal main
                return
            node = Node(self.ctr, func, parent_node)
            self.ctr += 1
            parent_node.children.append(node)
        else:
            node = Node(self.ctr, func)
            self.ctr += 1
            self.root = node
        return

    def delete(self, node: int) -> None:
        """Deletes the given Node, children are removed

        Parameters
        ----------
        node : int
            Node to be deleted

        Returns
        -------
        None
        """
        node = self.search(node)
        node.parent.children.remove(node)
        return

    def replace(self, node: int, func: 'function') -> None:
        """Replaces the function of the given Node with func

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
         """ Formats the tree into a string, so we can save the trees state easily

        Parameters
        ----------
        node : Node
            iterates through nodes of tree

        Returns
        -------
        None
        """       
        
        if node == None:
            return
        self.string = self.string + self.root.func + self.root.num
        for child in node.children:
            self.serialize(child, self.string)
        self.string = self.string + ')'
        return

    def deserialize(self, node, saved_string) -> int:
        """ Returns the serialized string back to a tree

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

        if saved_string[0] = ')':
            return 1
        
        string = string + saved_string[0]
        self.string = self.string[1:]

        node = Node(int(string[0]), func_list[string[1]])
        # references dictionary of functions ( keyed with letters)

        for child in node.children:
            if self.deserialize(child, saved_string):
                break
        return 0


    def save_tree(self) -> None:
        """Saves the current tree state as a pre-ordered list

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        self.serialize(self.root)
        self.saved_states.append(self.string)
        # save the serialized string
        self.string = ""
        # reset the trees string for the next save
        return

    def restore_tree(self, state_number) -> None:
        """Saves the current tree state as a pre-ordered list

        Parameters
        ----------
        state_number : int
            version of saved tree that we will restore

        Returns
        ----------
        None
        """  

        self.deserialize(self.root, self.saved_state[state_number])
        # the tree is saved here from the root

        return


    def _preorder(self, curr: Node, result: list) -> list[int]:
        """ stores the tree as a list of nodes in preorder
            
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

    def traverse(self) -> list[int]:
        """Returns a preorder traversal of the Tree as a list

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

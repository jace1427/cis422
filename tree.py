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
        self.ctr = 0

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

    def _preorder(self, curr: Node, result: list) -> list[int]:
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

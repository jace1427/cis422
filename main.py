"""Tree UI Program Driver

Author:

    Justin Spidell
"""
import sys
from tree import Tree
from funcs import *


def help() -> None:
    """
    Displays all possible commands while in the main loop

        Parameters:
            None

        Returns:
            None
    """
    sys.stdout.write("\nHelp:\n")
    sys.stdout.write("e or edit: edit a tree\n")
    sys.stdout.write("l or list: list trees\n")
    sys.stdout.write("n or new: create new tree\n")
    sys.stdout.write("h or help: help\n")
    sys.stdout.write("q or quit: exit program\n\n")


def edit_help() -> None:
    """
    Displays all possible commands while editing a tree

        Parameters:
            None

        Returns:
            None
    """
    sys.stdout.write("\nEdit Help:\n")
    sys.stdout.write("a or add: add a node\n")
    sys.stdout.write("d or del: delete a node\n")
    sys.stdout.write("q or quit: quit to main menu\n\n")


def new() -> None:
    """
    Creates a new Tree

        Parameters:
            None

        Returns:
            None
    """
    sys.stdout.write("New Tree name: ")
    name = input()
    name.strip()
    t = Tree(name)
    trees[name] = t
    edit(t)


def get_tree(name="") -> None:
    """
    Prompts the user for a tree to edit, then calls edit on that tree

        Parameters:
            name (string): the name of the new tree

        Returns:
            None
    """
    if name == "":
        sys.stdout.write("\nChoose a Tree to edit:\n")

        # Print tree names
        tree_list = list(trees.keys())
        for i in range(len(tree_list)):
            sys.stdout.write(str(i) + ". " + tree_list[i] + "\n")
        sys.stdout.write("\n")

        # Prmopt for input
        sys.stdout.write("Tree name: ")
        name = input()
        name.strip()

        # Saftey check
        if trees.get(name) is None:
            sys.stdout.write(f"Tree ({name}) does not exist")
            # Recursive call if failed
            get_tree()
        else:
            # Preperation for edit call
            t = trees.get(name)

    else:
        # Saftey check
        if trees.get(name) is None:
            sys.stdout.write(f"Tree ({name}) does not exist")
            # Recursive call if failed
            get_tree()
        else:
            # Preperation for edit call
            t = trees.get(name)

    edit(t)


def add(tree: Tree) -> None:
    """
    TODO

        Parameters:
            tree (Tree): A tree that will have a node added

        Returns:
            None
    """
    return


def delete(tree: Tree) -> None:
    """
    TODO

        Parameters:
            tree (Tree): A tree that will have a node deleted

        Returns:
            None
    """
    return


def edit(tree: Tree) -> None:
    """
    Description of the function

        Parameters:
            tree (Tree): A tree to be edited

        Returns:
            None
    """
    sys.stdout.write(f"\nEditing {tree.name}, type h for help\n")

    # Edit while loop
    while True:
        sys.stdout.write(f"(Editing {tree.name})Enter a command: ")
        cmd = input()
        cmd.strip().lower()

        if cmd == "a" or cmd == "add":
            add(tree)

        elif cmd == "d" or cmd == "del":
            delete(tree)

        elif cmd == "h" or cmd == "help":
            edit_help()

        elif cmd == "q" or cmd == "quit":
            sys.stdout.write("\n")
            break

        else:
            sys.stdout.write(f"{cmd} is not a command\n")


def list_trees() -> None:
    """
    Loops through and prints the name of each tree

        Parameters:
            None

        Returns:
            None
    """
    sys.stdout.write("\nTrees:\n")
    for t in list(trees.keys()):
        sys.stdout.write(trees[t].name + "\n")
    sys.stdout.write("\n")


def main() -> None:
    """
    Main program driver.

    Uses a while loop to prompt for input and calls
    respective functions
    """
    sys.stdout.write("Welcome to tree.py, type h for help\n")

    # Main while loop
    while True:
        sys.stdout.write("Enter a command: ")
        cmd = input()
        cmd.strip().lower()

        if cmd == "q" or cmd == "quit":
            break

        elif cmd == "h" or cmd == "help":
            help()

        elif cmd == "n" or cmd == "new":
            new()

        elif cmd == "e" or cmd == "edit:":
            get_tree()

        elif cmd == "l" or cmd == "list":
            list_trees()

        else:
            sys.stdout.write(f"{cmd} is not a command, type h for help\n")


if __name__ == '__main__':
    # dictionary of trees created
    trees = {}

    # possible functions
    funcs = [asdf, qwer, zxcv]

    main()

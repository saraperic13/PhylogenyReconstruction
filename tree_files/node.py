class Node:

    def __init__(self, name, ancestor=None, descendants=[]):
        self.name = name
        self.ancestor = ancestor
        self.descendants = descendants

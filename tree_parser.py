class Tree:

    def __init__(self, tips, nodes, edges):
        self.tips = tips
        self.nodes = nodes
        self.edges = edges


class Tip:

    def __init__(self, name, parent):
        self.name = name
        self.parent = parent


class Node(Tip):

    def __init__(self, name, parent, encoded_sequence):
        super().__init__(name, parent)
        self.encoded_sequence = encoded_sequence


class Edge:

    def __init__(self, tip, node):
        self.tip = tip
        self.node = node


def parse(file_name):
    pass


def are_together(node_1, node_2):
    pass

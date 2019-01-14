import random


class Tree:

    def __init__(self, leaves):
        self.nodes = []
        self.leaves = leaves
        self.number_of_named = len(leaves)

    def add_node(self, node):
        self.nodes.append(node)

    def get_number_of_named(self):
        return self.number_of_named

    def get_number_of_leaves(self):
        return len(self.leaves)

    def increment_number_of_named(self):
        self.number_of_named += 1

    def get_random_node(self):
        subroot = random.choice(self.nodes)
        while not subroot.descendants:
            subroot = random.choice(self.nodes)

        return subroot

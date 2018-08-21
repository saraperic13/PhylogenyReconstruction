import random
import numpy as np


class Tree:

    def __init__(self, number_of_named, nodes=[]):
        self.nodes = nodes
        self.number_of_named = number_of_named

    def add_node(self, node):
        self.nodes.append(node)

    def get_number_of_named(self):
        return self.number_of_named

    def increment_number_of_named(self):
        self.number_of_named += 1

    # def __str__(self):
    #     for node in self.nodes:
    #         print("Children of node ", node.name, [n.name for n in node.descendants])

    def get_random_node(self):
        subroot = random.choice(self.nodes)
        while not subroot.descendants:
            subroot = random.choice(self.nodes)
        return subroot

    def get_all_descendants(self, node, descendants):
        if not node.descendants:
            return
        for item in node.descendants:
            descendants.append(item)
        for child in node.descendants:
            self.get_all_descendants(child, descendants)

    def get_random_descendants_from_subroot(self, subroot):
        descendants = []
        self.get_all_descendants(subroot, descendants)
        return descendants if not descendants else random.choice(descendants), random.choice(descendants)

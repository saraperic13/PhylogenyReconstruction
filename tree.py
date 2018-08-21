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

    def __str__(self):
        for node in self.nodes:
            print("Children of node ", node.name, [n.name for n in node.descendants])

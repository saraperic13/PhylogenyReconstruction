from newick import load
import io
from tree import Tree


def bfs(node, tree):

    levels = [1]
    node.level = 0
    queue = [node]

    while queue:

        vertex = queue.pop(0)

        if vertex not in tree.nodes:

            if not vertex.name:
                tree.increment_number_of_named()
                vertex.name = tree.get_number_of_named()

            tree.add_node(vertex)

            if vertex.level + 1  == len(levels):
                levels.append(0)
            levels[vertex.level + 1] += len(vertex.descendants)

            tree.levels.append(len(vertex.descendants))
            for child in vertex.descendants:

                queue.append(child)
                child.level = vertex.level + 1

    tree.levels = [y for y in levels if y != 0]




# def load_tree(node, tree):
#     if not node.name:
#         tree.increment_number_of_named()
#         node.name = tree.get_number_of_named()
#
#     tree.add_node(node)
#
#     for child in node.descendants:
#         load_tree(child, tree)


def parse(file_name):
    with io.open(file_name, encoding='utf8') as fp:
        loaded_tree = load(fp)[0]
        tree = Tree(len(loaded_tree.get_leaves()))
        bfs(loaded_tree, tree)
    return tree


parse("dataset/phylogeny.tree")
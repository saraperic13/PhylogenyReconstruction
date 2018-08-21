from newick import load
import io
from tree import Tree


def load_tree(node, tree):
    if not node.name:
        tree.increment_number_of_named()
        node.name = tree.get_number_of_named()

    tree.add_node(node)
    print("ANCESTOR ", node.ancestor)

    for child in node.descendants:
        load_tree(child, tree)


def parse(file_name):
    with io.open(file_name, encoding='utf8') as fp:
        loaded_tree = load(fp)[0]
        tree = Tree(len(loaded_tree.get_leaves()))
        load_tree(loaded_tree, tree)
    return tree


def get_first_ancestor_after_root(node, root):
    if node.ancestor == root:
        return node
    return get_first_ancestor_after_root(node.ancestor, root)


def are_together(node_1, node_2, root):
    ancestor_node_1 = get_first_ancestor_after_root(node_1, root)
    ancestor_node_2 = get_first_ancestor_after_root(node_2, root)

    if ancestor_node_1 == ancestor_node_2:
        return True
    return False


def main():
    tree = parse('dataset/small_tree.tree')
    print(tree)
    subroot = tree.get_random_node()



main()

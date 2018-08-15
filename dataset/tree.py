from newick import load
import io


def recursion(node):
    if node.descendants:
        print("Moji potomci ", node.name, [n.name for n in node.descendants])
    for child in node.descendants:
        recursion(child)


with io.open('phylogeny.tree', encoding='utf8') as fp:
    trees = load(fp)
    recursion(trees[0])

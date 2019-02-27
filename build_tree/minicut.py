import networkx as nx


def get_partition(left_nodes, right_nodes, are_together):
    graph = nx.Graph()
    for i in range(len(left_nodes)):
        graph.add_edge(left_nodes[i], right_nodes[i], weight=are_together[i][0])
    _, partition = nx.stoer_wagner(graph)
    return partition


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edge('1', '2', weight=.8)
    G.add_edge('1', '3', weight=.2)
    G.add_edge('1', '4', weight=.4)
    G.add_edge('2', '3', weight=.2)
    G.add_edge('2', '4', weight=.4)
    G.add_edge('3', '4', weight=.9)
    cut_value, partition = nx.stoer_wagner(G)

    print(cut_value)
    print(partition)

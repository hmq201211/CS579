import networkx as nx


def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    neighbors = graph.neighbors(node)
    unique_neighbor_nodes = set(neighbors)
    denominator_node = 0.0
    for node_neighbors_node in unique_neighbor_nodes:
        denominator_node += graph.degree(node_neighbors_node)
    result_list = []
    for candidate in graph.nodes:
        if candidate not in unique_neighbor_nodes and candidate != node:
            candidate_neighbors = set(graph.neighbors(candidate))
            node_candidate_intersections = unique_neighbor_nodes & candidate_neighbors
            numerator = 0.0
            for intersection_node in node_candidate_intersections:
                numerator += 1. / graph.degree(intersection_node)
            denominator_candidate = 0.0
            for candidate_neighbor_node in candidate_neighbors:
                denominator_candidate += graph.degree(candidate_neighbor_node)
            score = numerator / (1. / denominator_candidate + 1. / denominator_node)
            result_list.append(((node, candidate), score))
    return result_list


# def main():
#     g = nx.Graph()
#     g.add_edges_from(
#         [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
#     temp = jaccard_wt(g, 'B')
#     print(temp)
#
#
# if __name__ == '__main__':
#     main()

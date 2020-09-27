"""
Cluster data.
"""
from collections import Counter, defaultdict, deque
import copy
from itertools import combinations
import math
import networkx as nx
import urllib.request
import json
import networkx as nx
from collections import Counter, defaultdict, deque
import sys
import glob
import os
import time
import matplotlib.pyplot as plt
import scipy


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    counter = Counter()
    for user in users:
        counter.update(user['friends'])
    return counter


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.Graph()
    for user in users:
        for friend in user['friends']:
            if friend_counts[friend] > 1:
                graph.add_edge(user['screen_name'], str('user: ' + str(friend)))
    return graph


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    plt.figure(frameon=True)
    plt.axis('off')
    plt.title('Twitter Network Graph')
    candidate_labels = {}
    for user in users:
        candidate_labels.update({user['screen_name']: user['screen_name']})
    # print(candidate_labels)
    nx.draw_networkx(graph, node_size=30, node_color='blue', alpha=0.5, edge_color='red', labels=candidate_labels,
                     font_size=10, font_color='black')
    plt.savefig(filename)


def get_components(graph):
    """
    A helper function you may use below.
    Returns the list of all connected components in the given graph.
    """
    return [c for c in nx.connected_component_subgraphs(graph)]


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    node_degrees = graph.degree()
    graph_copy = graph.copy()
    node_list_to_remove = []
    for node in graph_copy.node():
        if node_degrees[node] < min_degree:
            node_list_to_remove.append(node)
    graph_copy.remove_nodes_from(node_list_to_remove)
    return graph_copy


def load_data(path):
    return [json.load(open(f)) for f in glob.glob(os.path.join(path, '*.txt'))]


def save_data(conclusion, path):
    with open(path, 'w', encoding='utf-8') as writer:
        writer.write(conclusion)
    writer.close()
    # print(conclusion)
    # json.dump(conclusion, open(path, 'w'))


def girvan_newman(G, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html

    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """
    min_degree = 10
    max_degree = 30
    if G.order() < min_degree:
        return None
    elif G.order() in range(min_degree, max_degree):
        return [G.nodes()]

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        # print(indent + 'removing ' + str(edge_to_remove))
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = []
    # print(indent + 'components=' + str(result))
    for c in components:
        cluster = girvan_newman(c, depth + 1)
        if cluster is not None:
            result.extend(cluster)

    return result


def main():
    print("begin to load data")
    users = load_data("./shared/cluster_data/")
    print("data loaded")
    friend_counts = count_friends(users)
    print("begin to crate graph")
    graph = create_graph(users, friend_counts)
    print("graph created")
    print("begin to draw graph")
    draw_network(graph, users, './shared/cluster_data/pic')
    print("draw graph finished")
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    print("begin to get subgraph with certain degree")
    subgraph = get_subgraph(graph, 3)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print("subgraph task finished")
    print("begin to use girvan_newman to perform community detection")
    clusters = girvan_newman(subgraph)
    print("cluster task finished, there are the clusters:")
    all_node = 0
    for cluster in clusters:
        all_node += len(cluster)
        print(cluster, len(cluster))
    conclusion = 'There are ' + str(
        len(clusters)) + " clusters discovered. \nThe average user count per community is " + str(
        all_node / len(clusters)) + "."
    print("begin to save cluster result")
    save_data(conclusion, './shared/summary_data/cluster_result.txt')
    print("cluster result saved")


if __name__ == "__main__":
    main()

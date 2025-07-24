from typing import Any, Callable, Set
import networkx as nx
from collections import deque


def bfs_mv(G, P, v, t):
    """
    Procedure BFS_MV as described in Figure 3 of Di Stefano's paper.
    Calculates distances from a starting vertex v to all other vertices,
    and specifically to points in P.

    Parameters:
    - G (networkx.Graph): The connected graph.
    - P (set): A set of "points" (vertices) in the graph.
    - v (node): The starting vertex for BFS.
    - t (bool): If True, distances are calculated in G.
                If False, distances are calculated in G excluding other points in P (P \\ {u,v}).

    Returns:
    - dict: DP (distance to points in P), where DP[p] is the distance from v to p.
    """

    D = {node: float("inf") for node in G.nodes()}
    DP = {p_node: float("inf") for p_node in P}

    D[v] = 0
    if v in P:
        DP[v] = 0

    Q = deque()
    Q.append(v)

    while Q:
        u = Q.popleft()

        for w in G.neighbors(u):
            if D[w] == float("inf"):
                D[w] = D[u] + 1
                if w in P:
                    DP[w] = D[w]
                if t or w not in P:
                    Q.append(w)

    return DP


def mv(G, P):
    """
    Procedure MV as described in Figure 4 of Di Stefano's paper.
    Checks if a given set P is a mutual-visibility set in graph G.

    Parameters:
    - G (networkx.Graph): The undirected graph.
    - P (set): The set of "points" (vertices) to check for mutual visibility.

    Returns:
    - bool: True if P is a mutual-visibility set, False otherwise.
    """

    if not P or len(P) == 1:
        return True

    first_point = next(iter(P))
    connected_component_of_first_point = None

    for component in nx.connected_components(G):
        if first_point in component:
            connected_component_of_first_point = component
            break

    if connected_component_of_first_point is None:
        return False

    for point in P:
        if point not in connected_component_of_first_point:
            return False

    H = G.subgraph(connected_component_of_first_point)
    for p in P:

        distances_in_H = bfs_mv(H, P, p, True)
        distances_in_H_minus_p = bfs_mv(H, P, p, False)
        if distances_in_H != distances_in_H_minus_p:
            return False

    return True

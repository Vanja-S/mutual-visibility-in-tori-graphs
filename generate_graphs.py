import networkx as nx
from typing import Union


def generate_torus(m: int, n: int) -> nx.Graph:
    """
    Generate a torus graph T_{m,n} = C_m □ C_n (Cartesian product of two cycles).

    Parameters:
    -----------
    m : int
        Size of the first cycle C_m
    n : int
        Size of the second cycle C_n

    Returns:
    --------
    nx.Graph
        The torus graph T_{m,n}
    """
    if m < 3 or n < 3:
        raise ValueError("Both m and n must be at least 3 for valid cycle graphs")

    # Create the torus graph
    torus = nx.Graph()

    # Add all vertices (i, j) where i ∈ {0, 1, ..., m-1} and j ∈ {0, 1, ..., n-1}
    vertices = [(i, j) for i in range(m) for j in range(n)]
    torus.add_nodes_from(vertices)

    # Add edges according to Cartesian product definition
    for i in range(m):
        for j in range(n):
            current_vertex = (i, j)

            # Edges from C_m component (j stays same, i changes to adjacent in cycle)
            # In C_m, vertex i is adjacent to (i-1) mod m and (i+1) mod m
            next_i = (i + 1) % m
            prev_i = (i - 1) % m

            torus.add_edge(current_vertex, (next_i, j))
            torus.add_edge(current_vertex, (prev_i, j))

            # Edges from C_n component (i stays same, j changes to adjacent in cycle)
            # In C_n, vertex j is adjacent to (j-1) mod n and (j+1) mod n
            next_j = (j + 1) % n
            prev_j = (j - 1) % n

            torus.add_edge(current_vertex, (i, next_j))
            torus.add_edge(current_vertex, (i, prev_j))

    return torus


def generate_generalised_petersen(n: int, k: int) -> nx.Graph:
    """
    Generate a generalized Petersen graph G(n,k).

    The graph has vertex set {u_0, u_1, ..., u_{n-1}, v_0, v_1, ..., v_{n-1}}
    and edge set {u_i u_{i+1}, u_i v_i, v_i v_{i+k} | 0 ≤ i ≤ n-1}
    where subscripts are read modulo n.

    Parameters:
    -----------
    n : int
        Number of vertices in each part (outer and inner cycles)
    k : int
        Step size for the inner connections, must satisfy k < n/2

    Returns:
    --------
    nx.Graph
        The generalized Petersen graph G(n,k)

    Raises:
    -------
    ValueError
        If k >= n/2 or if n < 3 or k < 1
    """
    # Input validation
    if n < 3:
        raise ValueError("n must be at least 3 for a valid graph")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k >= n / 2:
        raise ValueError(f"k must be less than n/2. Got k={k}, n/2={n/2}")

    # Create the graph
    G = nx.Graph()

    # Add vertices
    # Outer vertices: u_0, u_1, ..., u_{n-1}
    outer_vertices = [f"u_{i}" for i in range(n)]
    # Inner vertices: v_0, v_1, ..., v_{n-1}
    inner_vertices = [f"v_{i}" for i in range(n)]

    G.add_nodes_from(outer_vertices)
    G.add_nodes_from(inner_vertices)

    # Add edges according to the definition
    for i in range(n):
        # Edge type 1: u_i u_{i+1} (outer cycle)
        # Connect each outer vertex to the next one (modulo n)
        next_outer = (i + 1) % n
        G.add_edge(f"u_{i}", f"u_{next_outer}")

        # Edge type 2: u_i v_i (spokes)
        # Connect each outer vertex to corresponding inner vertex
        G.add_edge(f"u_{i}", f"v_{i}")

        # Edge type 3: v_i v_{i+k} (inner connections)
        # Connect each inner vertex to the vertex k steps away (modulo n)
        next_inner = (i + k) % n
        G.add_edge(f"v_{i}", f"v_{next_inner}")

    return G

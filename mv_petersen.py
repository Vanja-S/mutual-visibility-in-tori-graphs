from itertools import combinations
import networkx as nx
import csv
import json
from datetime import datetime
from typing import Set, List, Tuple, Optional

from bfs_mv import mv
from generate_graphs import generate_generalised_petersen


def bruteforce_mv_petersen_random(G, n: int, k: int) -> Tuple[int, Set]:
    """
    Random combinations search for the maximum mutual-visibility set in a generalized Petersen graph.
    Similar to the optimized torus implementation.

    Parameters:
    -----------
    G : networkx.Graph
        The generalized Petersen graph to analyze
    n : int
        Parameter n of the generalized Petersen graph G(n,k)
    k : int
        Parameter k of the generalized Petersen graph G(n,k)

    Returns:
    --------
    Tuple[int, Set]
        A tuple containing the maximum cardinality and the actual set
    """
    vertices = list(G.nodes())
    max_cardinality = 0
    best_set = set()

    # Upper bound for search - similar to torus approach
    max_size_to_check = min(len(vertices), 2 * n)  # Heuristic upper bound

    print(f"Using random combinations search up to size {max_size_to_check}...")
    print(f"Total vertices in graph: {len(vertices)}")

    # Check subsets of increasing size
    for size in range(1, max_size_to_check + 1):
        print(f"Checking subsets of size {size}...")

        found_mv_set = False
        subset_count = 0

        # Generate all subsets of current size
        for subset in combinations(vertices, size):
            subset_count += 1
            subset_set = set(subset)

            # Check if this subset is a mutual-visibility set
            if mv(G, subset_set):
                found_mv_set = True
                max_cardinality = size
                best_set = subset_set
                print(f"Found MV set of size {size}: {subset_set}")
                break  # Exit the inner loop immediately - OPTIMIZATION

        print(f"  Checked {subset_count} subsets of size {size}")

        # If no mutual-visibility set of current size exists,
        # no larger ones will exist either
        if not found_mv_set:
            print(f"No MV sets of size {size} found. Stopping search.")
            break

    return max_cardinality, best_set


def bruteforce_mv_petersen_backtrack(G, n: int, k: int) -> Tuple[int, Set]:
    """
    Backtracking search for the maximum mutual-visibility set in a generalized Petersen graph.

    Parameters:
    -----------
    G : networkx.Graph
        The generalized Petersen graph to analyze
    n : int
        Parameter n of the generalized Petersen graph G(n,k)
    k : int
        Parameter k of the generalized Petersen graph G(n,k)

    Returns:
    --------
    Tuple[int, Set]
        A tuple containing the maximum cardinality and the actual set
    """
    vertices = list(G.nodes())
    max_cardinality = 0
    best_set = set()

    # Upper bound for search - you can adjust this based on your computational resources
    # For Petersen graphs, we might expect smaller MV sets than torus graphs
    max_size_to_check = min(len(vertices), 2 * n)  # Heuristic upper bound

    print(f"Using backtracking search up to size {max_size_to_check}...")
    print(f"Total vertices in graph: {len(vertices)}")

    def backtrack(
        current_set: Set, remaining_vertices: List, start_idx: int, target_size: int
    ) -> bool:
        """
        Backtracking function to find MV sets of a specific target size.
        Returns True if a valid MV set of target_size is found.
        """
        nonlocal max_cardinality, best_set

        # If we've reached the target size, check if it's a valid MV set
        if len(current_set) == target_size:
            if mv(G, current_set):
                if target_size > max_cardinality:
                    max_cardinality = target_size
                    best_set = current_set.copy()
                    print(f"Found MV set of size {target_size}: {current_set}")
                return True
            return False

        # If we can't possibly reach target_size with remaining vertices, prune
        if len(current_set) + (len(remaining_vertices) - start_idx) < target_size:
            return False

        # Try adding each remaining vertex
        for i in range(start_idx, len(remaining_vertices)):
            vertex = remaining_vertices[i]
            current_set.add(vertex)

            # Recursive call
            if backtrack(current_set, remaining_vertices, i + 1, target_size):
                return True  # Found a valid set, can stop searching this size

            # Backtrack
            current_set.remove(vertex)

        return False

    # Search for MV sets of increasing size using backtracking
    for size in range(1, max_size_to_check + 1):
        print(f"Searching for MV sets of size {size}...")

        found = backtrack(set(), vertices, 0, size)

        if not found:
            print(f"No MV sets of size {size} found. Stopping search.")
            break

    return max_cardinality, best_set


def bruteforce_mv_petersen_iterative(G, n: int, k: int) -> Tuple[int, Set]:
    """
    Alternative iterative approach with early pruning for smaller graphs.
    """
    vertices = list(G.nodes())
    max_cardinality = 0
    best_set = set()

    # For very small graphs, use the original iterative approach
    max_size_to_check = min(len(vertices), 2 * n)

    print(f"Using iterative search up to size {max_size_to_check}...")

    for size in range(1, max_size_to_check + 1):
        print(f"Checking subsets of size {size}...")

        found_mv_set = False
        subset_count = 0
        max_subsets_to_check = 10000  # Limit to prevent excessive computation

        for subset in combinations(vertices, size):
            subset_count += 1
            subset_set = set(subset)

            if mv(G, subset_set):
                found_mv_set = True
                max_cardinality = size
                best_set = subset_set
                print(f"Found MV set of size {size}: {subset_set}")
                break

            # Early termination if too many subsets
            if subset_count >= max_subsets_to_check:
                print(
                    f"  Reached subset limit ({max_subsets_to_check}), moving to backtracking..."
                )
                return bruteforce_mv_petersen_backtrack(G, n, k)

        print(f"  Checked {subset_count} subsets of size {size}")

        if not found_mv_set:
            print(f"No MV sets of size {size} found. Stopping search.")
            break

    return max_cardinality, best_set


def analyze_petersen_range(
    k: int, min_n: int, max_n: int, output_file: str = None
) -> dict:
    """
    Analyze mutual-visibility sets for generalized Petersen graphs G(n,k) in a given range.

    Parameters:
    -----------
    k : int
        Fixed parameter k for all graphs G(n,k)
    min_n : int
        Minimum value of n to analyze
    max_n : int
        Maximum value of n to analyze (inclusive)
    output_file : str, optional
        CSV file to save results (auto-generated if None)

    Returns:
    --------
    dict
        Results dictionary with n values as keys
    """
    if output_file is None:
        output_file = f"petersen_G_n_{k}_mv_results.csv"

    results = {}

    # Prepare CSV file
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "n",
            "k",
            "vertices",
            "edges",
            "max_mv_size",
            "mv_set",
            "mv_set_readable",
            "search_method",
            "timestamp",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(
            f"Analyzing Generalized Petersen graphs G(n,{k}) for n = {min_n} to {max_n}"
        )
        print(f"Results will be saved to: {output_file}")

        for n in range(min_n, max_n + 1):
            # Check if G(n,k) is valid
            if k >= n / 2:
                print(f"\nSkipping G({n},{k}): k >= n/2")
                continue

            print(f"\n=== Analyzing G({n},{k}) ===")

            try:
                # Generate the Petersen graph
                petersen = generate_generalised_petersen(n, k)

                print(f"Graph properties:")
                print(f"  Vertices: {petersen.number_of_nodes()}")
                print(f"  Edges: {petersen.number_of_edges()}")

                # Choose search method based on graph size
                total_vertices = petersen.number_of_nodes()
                if total_vertices <= 12:  # Small graphs
                    max_mv_size, mv_set = bruteforce_mv_petersen_random(petersen, n, k)
                    search_method = "random"
                else:  # Larger graphs
                    max_mv_size, mv_set = bruteforce_mv_petersen_random(petersen, n, k)
                    search_method = "random"

                # Convert set to readable format
                mv_set_list = sorted(list(mv_set)) if mv_set else []
                mv_set_readable = ", ".join(mv_set_list) if mv_set_list else ""

                # Store results
                result_data = {
                    "n": n,
                    "k": k,
                    "vertices": petersen.number_of_nodes(),
                    "edges": petersen.number_of_edges(),
                    "max_mv_size": max_mv_size,
                    "mv_set": mv_set_list,  # For JSON serialization
                    "mv_set_readable": mv_set_readable,  # For CSV readability
                    "search_method": search_method,
                    "timestamp": datetime.now().isoformat(),
                }

                results[n] = result_data

                # Write to CSV
                writer.writerow(result_data)
                csvfile.flush()  # Ensure data is written immediately

                print(f"G({n},{k}): μ(G({n},{k})) = {max_mv_size}")
                if mv_set:
                    print(f"  MV set: {mv_set_readable}")

            except Exception as e:
                print(f"Error analyzing G({n},{k}): {e}")
                error_data = {
                    "n": n,
                    "k": k,
                    "vertices": 0,
                    "edges": 0,
                    "max_mv_size": -1,
                    "mv_set": [],
                    "mv_set_readable": f"ERROR: {str(e)}",
                    "search_method": "failed",
                    "timestamp": datetime.now().isoformat(),
                }
                results[n] = error_data
                writer.writerow(error_data)
                csvfile.flush()

    # Also save as JSON for easy loading
    json_file = output_file.replace(".csv", ".json")
    with open(json_file, "w") as jsonfile:
        json.dump(results, jsonfile, indent=2, default=str)

    print(f"\nResults saved to:")
    print(f"  CSV: {output_file}")
    print(f"  JSON: {json_file}")

    return results


# Example usage
if __name__ == "__main__":
    print("Starting analysis of Generalized Petersen graphs...")

    # Single k value analysis
    k = 1
    results_k2 = analyze_petersen_range(k, min_n=16, max_n=20)

    print("\nAnalysis complete!")

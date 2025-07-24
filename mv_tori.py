from itertools import combinations
import json
import csv
from datetime import datetime
from typing import List, Set, Tuple
import multiprocessing as mp
from functools import partial

from bfs_mv import mv
from generate_graphs import generate_torus


def check_subset_batch(args):
    """
    Helper function to check a batch of subsets for mutual-visibility.
    Used for parallel processing.

    Parameters:
    -----------
    args : tuple
        A tuple containing (G_edges, subset_batch) where:
        - G_edges: list of graph edges (to reconstruct the graph)
        - subset_batch: list of subsets to check

    Returns:
    --------
    tuple
        (found_mv_set, first_mv_set_found) where:
        - found_mv_set: boolean indicating if any MV set was found
        - first_mv_set_found: the first MV set found (or None)
    """
    import networkx as nx

    G_edges, subset_batch = args

    # Reconstruct the graph from edges
    G = nx.Graph()
    G.add_edges_from(G_edges)

    for subset in subset_batch:
        subset_set = set(subset)
        if mv(G, subset_set):
            return True, subset_set

    return False, None


def chunk_combinations(combinations_iter, chunk_size):
    """
    Generator to yield chunks of combinations.
    """
    chunk = []
    for combo in combinations_iter:
        chunk.append(combo)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def bruteforce_mv_torus_backtrack(G, m: int, n: int) -> Tuple[int, Set]:
    """
    Backtracking search for the maximum mutual-visibility set in a torus graph.

    Parameters:
    -----------
    G : networkx.Graph
        The torus graph to analyze
    m : int
        Size of the first cycle in the torus T_{m,n}
    n : int
        Size of the second cycle in the torus T_{m,n}

    Returns:
    --------
    Tuple[int, Set]
        A tuple containing the maximum cardinality and the actual set
    """
    vertices = list(G.nodes())
    max_cardinality = 0
    best_set = set()

    # Upper bound for search: 3 * min(m, n)
    max_size_to_check = 3 * min(m, n)

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
    for size in range(1, min(max_size_to_check + 1, len(vertices) + 1)):
        print(f"Searching for MV sets of size {size}...")

        found = backtrack(set(), vertices, 0, size)

        if not found:
            print(f"No MV sets of size {size} found. Stopping search.")
            break

    return max_cardinality, best_set


def bruteforce_mv_tori(G, m: int, n: int) -> int:
    """
    Brute force search for the maximum mutual-visibility set cardinality in a torus graph.
    Uses parallelization for larger graphs (m >= 12).

    Parameters:
    -----------
    G : networkx.Graph
        The torus graph to analyze
    m : int
        Size of the first cycle in the torus T_{m,n}
    n : int
        Size of the second cycle in the torus T_{m,n}

    Returns:
    --------
    int
        The maximum cardinality of mutual-visibility sets found
    """
    vertices = list(G.nodes())
    max_cardinality = 0

    # Upper bound for search: 3 * min(m, n)
    max_size_to_check = 3 * min(m, n)

    # Determine if we should use parallelization
    use_parallel = min(m, n) >= 6

    print(f"Searching subsets of size 1 to {max_size_to_check}...")
    print(f"Total vertices in graph: {len(vertices)}")
    if use_parallel:
        print(f"Using parallel processing (m={m}, n={n} >= 6)")
        num_processes = mp.cpu_count()
        print(f"Using {num_processes} CPU cores")
    else:
        print(f"Using sequential processing (m={m}, n={n} < 6)")

    # Check subsets of increasing size
    for size in range(1, min(max_size_to_check + 1, len(vertices) + 1)):
        print(f"Checking subsets of size {size}...")

        found_mv_set = False
        subset_count = 0

        if use_parallel and size > 5:  # Only parallelize for larger subset sizes
            # Parallel processing
            G_edges = list(G.edges())
            chunk_size = max(100, 1000 // size)  # More conservative chunk size

            # Use fewer processes to avoid memory issues
            num_processes_to_use = min(mp.cpu_count(), 8)

            with mp.Pool(processes=num_processes_to_use) as pool:
                # Create chunks of combinations
                combo_chunks = list(
                    chunk_combinations(combinations(vertices, size), chunk_size)
                )
                subset_count = sum(len(chunk) for chunk in combo_chunks)

                print(
                    f"  Processing {subset_count} subsets in {len(combo_chunks)} chunks using {num_processes_to_use} processes..."
                )

                # Prepare arguments for parallel processing
                args_list = [(G_edges, chunk) for chunk in combo_chunks]

                # Process chunks in parallel
                try:
                    results = pool.map(check_subset_batch, args_list)

                    # Check if any worker found an MV set
                    for found, mv_set in results:
                        if found:
                            found_mv_set = True
                            max_cardinality = size
                            print(f"Found MV set of size {size} - moving to next size")
                            break

                except KeyboardInterrupt:
                    print("Interrupted by user")
                    pool.terminate()
                    pool.join()
                    break
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    print("Falling back to sequential processing for this size...")
                    # Fall back to sequential processing
                    for subset in combinations(vertices, size):
                        subset_count += 1
                        subset_set = set(subset)
                        if mv(G, subset_set):
                            found_mv_set = True
                            max_cardinality = size
                            print(f"Found MV set of size {size} - moving to next size")
                            break
        else:
            # Sequential processing (original code)
            for subset in combinations(vertices, size):
                subset_count += 1
                subset_set = set(subset)

                # Check if this subset is a mutual-visibility set
                if mv(G, subset_set):
                    found_mv_set = True
                    max_cardinality = size
                    print(f"Found MV set of size {size} - moving to next size")
                    break

        print(f"  Checked {subset_count} subsets of size {size}")

        # If no mutual-visibility set of current size exists,
        # no larger ones will exist either (this is an optimization assumption)
        if not found_mv_set and size > 1:
            print(f"No MV sets of size {size} found. Stopping search.")
            break

    return max_cardinality


def analyze_square_torus_range(
    min_m: int, max_m: int, output_file: str = "torus_mv_results.csv"
) -> dict:
    """
    Analyze mutual-visibility sets for square torus graphs T_{m,m} in a given range.

    Parameters:
    -----------
    min_m : int
        Minimum dimension to analyze
    max_m : int
        Maximum dimension to analyze (inclusive)
    output_file : str
        CSV file to save results

    Returns:
    --------
    dict
        Results dictionary with dimensions as keys
    """
    results = {}

    # Prepare CSV file
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "dimension",
            "vertices",
            "edges",
            "max_mv_size",
            "search_limit",
            "timestamp",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Analyzing square torus graphs T_{{m,m}} for m = {min_m} to {max_m}")
        print(f"Results will be saved to: {output_file}")

        for m in range(min_m, max_m + 1):
            print(f"\n=== Analyzing T_{{{m},{m}}} ===")

            try:
                # Generate the torus graph
                torus = generate_torus(m, m)

                print(f"Graph properties:")
                print(f"  Vertices: {torus.number_of_nodes()}")
                print(f"  Edges: {torus.number_of_edges()}")
                print(f"  Search limit: {3 * min(m, m)} = {3 * m}")

                # Find maximum mutual-visibility set cardinality
                max_mv_size = bruteforce_mv_tori(torus, m, m)

                # Store results
                result_data = {
                    "dimension": m,
                    "vertices": torus.number_of_nodes(),
                    "edges": torus.number_of_edges(),
                    "max_mv_size": max_mv_size,
                    "search_limit": 3 * m,
                    "timestamp": datetime.now().isoformat(),
                }

                results[m] = result_data

                # Write to CSV
                writer.writerow(result_data)
                csvfile.flush()  # Ensure data is written immediately

                print(f"T_{{{m},{m}}}: μ(T_{{{m},{m}}}) = {max_mv_size}")

            except Exception as e:
                print(f"Error analyzing T_{{{m},{m}}}: {e}")
                error_data = {
                    "dimension": m,
                    "vertices": 0,
                    "edges": 0,
                    "max_mv_size": -1,
                    "search_limit": 3 * m,
                    "timestamp": datetime.now().isoformat(),
                }
                results[m] = error_data
                writer.writerow(error_data)
                csvfile.flush()

    # Also save as JSON for easy loading
    json_file = output_file.replace(".csv", ".json")
    with open(json_file, "w") as jsonfile:
        json.dump(results, jsonfile, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV: {output_file}")
    print(f"  JSON: {json_file}")

    return results


# Example usage
if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Start method already set

    # Test with a smaller range first, then adjust for your needs
    # For T_{15,15}, use: results = analyze_square_torus_range(15, 15, "torus_mv_results_15.csv")
    results = analyze_square_torus_range(6, 7, "torus_mv_results.csv")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"{'Dimension':<10} {'μ(T_m,m)':<10} {'Vertices':<10} {'Ratio':<8}")
    print("-" * 40)

    for m, data in results.items():
        if data["max_mv_size"] > 0:
            ratio = data["max_mv_size"] / data["vertices"]
            print(
                f"{m:<10} {data['max_mv_size']:<10} {data['vertices']:<10} {ratio:.3f}"
            )

import os
import json
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


def find_result_files(directory):
    """
    Find all result JSON files in the directory that match the pattern 'combined_algorithms_results_<number>.json'.

    :param directory: Directory to search for JSON files.
    :return: List of tuples containing the file path and iteration number.
    """
    result_files = []
    for file_name in os.listdir(directory):
        if file_name.startswith("combined_algorithms_results_") and file_name.endswith(".json"):
            try:
                num_iterations = int(file_name.split("_")[-1].split(".")[0])
                result_files.append((os.path.join(directory, file_name), num_iterations))
            except ValueError:
                continue
    return sorted(result_files, key=lambda x: x[1])


def extract_scores(file_path):
    """
    Extract unique scores from the result JSON file, skipping redundant sequences.

    :param file_path: Path to the JSON file.
    :return: Dictionary of algorithm scores or an empty dictionary if JSON is invalid.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: {file_path} is empty or contains invalid JSON. {e}")
        return {}

    unique_alignments = set()
    filtered_results = {}

    for algorithm, iterations in data.items():
        if not isinstance(iterations, list):
            print(f"Warning: {algorithm} has invalid data format. Skipping.")
            continue

        filtered_scores = []
        for result in iterations:
            if not isinstance(result, dict):
                print(f"Warning: Skipping malformed result entry: {result}")
                continue

            alignment = result.get("alignment", {})
            if not isinstance(alignment, dict):
                print(f"Warning: Skipping result with invalid alignment: {alignment}")
                continue

            align_key = (alignment.get("align1"), alignment.get("align2"))
            if align_key not in unique_alignments:
                unique_alignments.add(align_key)
                filtered_scores.append(result.get("score", 0))

        filtered_results[algorithm] = filtered_scores

    return filtered_results


def plot_results(results, num_iterations, max_score, output_path):
    """
    Plot the results of the algorithms and save as an image with smooth curves.

    :param results: Dictionary of algorithm scores.
    :param num_iterations: Number of iterations for the x-axis.
    :param max_score: Maximum score for the y-axis.
    :param output_path: Path to save the plot image.
    """
    # Filter out algorithms with no valid scores
    valid_scores = [scores for scores in results.values() if isinstance(scores, list) and scores]
    if not valid_scores:
        print("Warning: No valid scores to plot. Skipping this file.")
        return

    # Determine the overall min and max scores across all algorithms
    min_score = min(min(scores) for scores in valid_scores)
    max_score = max(max(scores) for scores in valid_scores)

    plt.figure(figsize=(10, 6))
    x_values = np.array(range(1, num_iterations + 1))

    for algorithm, scores in results.items():
        if isinstance(scores, list) and scores:
            # Trim or pad scores to the maximum number of iterations
            scores = scores[:num_iterations] + [0] * (num_iterations - len(scores))
            if len(scores) > 3:  # Use smoothing only if enough points are available
                x_smooth = np.linspace(x_values.min(), x_values.max(), 200)
                y_smooth = make_interp_spline(x_values, scores, k=3)(x_smooth)
                plt.plot(x_smooth, y_smooth, label=algorithm)
            else:
                plt.plot(x_values, scores, label=algorithm, marker="o")

    plt.xlabel("Iterations")
    plt.ylabel("Scores")
    plt.title("Algorithm Performance")
    plt.legend()

    # Dynamically adjust y-axis limits with padding
    y_padding = (max_score - min_score) * 0.1 if max_score != min_score else 5
    plt.ylim(min_score - y_padding, max_score + y_padding)

    plt.xlim(1, num_iterations)  # Set x-axis to the detected maximum number of iterations

    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    """
    Main function to generate graphs for all result files in the Comparison/Results directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")

    # Correct path for Results and Graphs directories
    results_dir = os.path.join(base_dir, "Results")  # Directly point to the correct Results folder
    graphs_dir = os.path.join(base_dir, "Graphs")

    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return

    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
        print(f"Created directory for graphs: {graphs_dir}")

    result_files = find_result_files(results_dir)

    if not result_files:
        print("No result files found in the Results directory.")
        return

    # Determine the maximum number of iterations from the file names
    max_iterations = max(num_iterations for _, num_iterations in result_files)
    print(f"Detected maximum number of iterations: {max_iterations}")

    for file_path, num_iterations in result_files:
        print(f"Processing file: {file_path} with {num_iterations} iterations")
        results = extract_scores(file_path)
        if not results:
            print(f"Skipping {file_path} due to empty or invalid results.")
            continue

        max_score = 0
        for scores in results.values():
            if isinstance(scores, list) and scores:
                max_score = max(max_score, max(scores))

        output_path = os.path.join(graphs_dir, f"algorithm_performance_{num_iterations}.png")
        plot_results(results, max_iterations, max_score, output_path)


if __name__ == "__main__":
    main()


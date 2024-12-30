import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
    Extract unique scores from the result JSON file, ensuring each iteration has a score.

    :param file_path: Path to the JSON file.
    :return: Dictionary of algorithm scores or an empty dictionary if JSON is invalid.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: {file_path} is empty or contains invalid JSON. {e}")
        return {}

    filtered_results = {}

    for algorithm, iterations in data.items():
        if not isinstance(iterations, list):
            print(f"Warning: {algorithm} has invalid data format. Skipping.")
            continue

        filtered_scores = [result.get("score", 0) for result in iterations if isinstance(result, dict)]
        filtered_results[algorithm] = filtered_scores

    return filtered_results

def plot_results(results, num_iterations, output_path):
    """
    Plot the results of the algorithms and save as an image with smooth curves.

    :param results: Dictionary of algorithm scores.
    :param num_iterations: Number of iterations for the x-axis.
    :param output_path: Path to save the plot image.
    """
    if not results:
        print("Warning: No valid scores to plot. Skipping this file.")
        return

    plt.figure(figsize=(12, 8))
    x_values = np.arange(1, num_iterations + 1)

    for algorithm, scores in results.items():
        if len(scores) != num_iterations:
            print(f"Warning: {algorithm} does not have scores for all {num_iterations} iterations. Padding with zeros.")
            scores = scores[:num_iterations] + [0] * (num_iterations - len(scores))

        # Smooth the curve
        x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
        y_smooth = make_interp_spline(x_values, scores, k=3)(x_smooth)
        plt.plot(x_smooth, y_smooth, label=algorithm)

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Scores", fontsize=14)
    plt.title("Algorithm Performance (Smooth Curves)", fontsize=16)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    """
    Main function to generate graphs for all result files in the Results directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "Results")
    graphs_dir = os.path.join(base_dir, "Graphs")

    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return

    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    result_files = find_result_files(results_dir)

    if not result_files:
        print("No result files found in the Results directory.")
        return

    for file_path, num_iterations in result_files:
        print(f"Processing file: {file_path} with {num_iterations} iterations")
        results = extract_scores(file_path)
        if not results:
            continue

        output_path = os.path.join(graphs_dir, f"algorithm_performance_{num_iterations}.png")
        plot_results(results, num_iterations, output_path)

if __name__ == "__main__":
    main()

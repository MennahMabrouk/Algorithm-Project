import os
import json
import numpy as np
from scipy.stats import ranksums


def calculate_metrics(result_dir, output_file):
    """
    Calculates the metrics specified:
    - Average minimum values for all algorithms.
    - Standard deviation for all algorithms and functions.
    - Elapsed time for computations.
    - P-value of Wilcoxon rank sum test for comparisons.

    :param result_dir: Directory containing the results JSON files.
    :param output_file: Path to save the generated output.
    """
    if not os.path.exists(result_dir):
        print(f"Error: The directory {result_dir} does not exist.")
        return

    all_results = {}
    for file_name in os.listdir(result_dir):
        if file_name.startswith("combined_algorithms_results_") and file_name.endswith(".json"):
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                for algo, results in data.items():
                    if algo not in all_results:
                        all_results[algo] = {"scores": [], "times": []}

                    for res in results:
                        all_results[algo]["scores"].append(res.get("score", 0))
                        all_results[algo]["times"].append(res.get("time", 0))

    avg_min_values = {}
    std_devs = {}
    elapsed_times = {}
    wilcoxon_p_values = {}

    # Compute average min values, standard deviation, and elapsed times
    for algo, data in all_results.items():
        scores = np.array(data["scores"])
        times = np.array(data["times"])

        avg_min_values[algo] = np.mean(scores)
        std_devs[algo] = np.std(scores)
        elapsed_times[algo] = np.sum(times)

    # Compute Wilcoxon rank sum test for comparisons
    algorithms = list(all_results.keys())
    comparisons_done = set()
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            if (algo1, algo2) not in comparisons_done and (algo2, algo1) not in comparisons_done:
                scores1 = np.array(all_results[algo1]["scores"])
                scores2 = np.array(all_results[algo2]["scores"])
                _, p_value = ranksums(scores1, scores2)
                wilcoxon_p_values[f"{algo1} vs {algo2}"] = p_value
                comparisons_done.add((algo1, algo2))

    # Write the results to a TXT file
    with open(output_file, "w") as f:
        f.write("Metrics Report\n")
        f.write("=" * 40 + "\n")
        f.write("Average Minimum Values:\n")
        for algo, value in avg_min_values.items():
            f.write(f"{algo}: {value}\n")
        f.write("\n")

        f.write("Standard Deviations:\n")
        for algo, value in std_devs.items():
            f.write(f"{algo}: {value}\n")
        f.write("\n")

        f.write("Elapsed Times:\n")
        for algo, value in elapsed_times.items():
            f.write(f"{algo}: {value}\n")
        f.write("\n")

        f.write("Wilcoxon Rank Sum Test P-Values:\n")
        for comparison, p_value in wilcoxon_p_values.items():
            f.write(f"{comparison}: {p_value}\n")
        f.write("\n")

    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    # Dynamically construct the paths using os.path.join
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Corrected base_dir
    results_directory = os.path.join(base_dir, "Comparison", "Results")
    output_filepath = os.path.join(results_directory, "metrics_report.txt")

    # Calculate and save the metrics
    calculate_metrics(results_directory, output_filepath)

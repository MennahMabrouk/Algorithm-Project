import os
import json
import numpy as np
from scipy.stats import ranksums


# Define benchmark functions
def F1(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def F2(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def F3(x):
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def F4(x):
    return np.sum(np.arange(1, len(x) + 1) * (x ** 4))

def F5(x):
    return np.sum(x ** 2)

def F6(x):
    return np.sum((x + 0.5) ** 2)

def F7(x):
    return -1 - np.cos(12 * np.sqrt(np.sum(x ** 2))) / (0.5 * np.sum(x ** 2) + 2)

def F8(x):
    return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2))

def F9(x):
    return np.sum(np.arange(1, len(x) + 1) * x ** 2)

def F10(x):
    return np.sum((x[:-1] - 1) ** 2 + np.arange(1, len(x)) * (2 * x[1:] ** 2 - x[:-1]) ** 2)

def F11(x):
    return np.sum(x) + np.prod(x)

def F12(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (x[0] ** 6) / 3 + x[0] * x[1] + 4 * x[1] ** 2 - 4 * x[1] ** 4

def F13(x):
    return np.max(np.abs(x))

def F14(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def F15(x):
    return np.sum(np.sin(x) * (np.sin(np.arange(1, len(x) + 1) * x ** 2 / np.pi)) ** 20)

def F16(x):
    return (
        1
        + (x[0] + x[1] + 1) ** 2
        * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)
        * (30 + (2 * x[0] - 3 * x[1]) * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    )

def F17(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )

def F18(x):
    return np.sum(x ** 2) ** 2

def F19(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x ** 2)))

def F20(x):
    return -np.exp(-np.sum(x ** 2)) * np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))))

# Map functions for easier use
benchmark_functions = {
    "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6, "F7": F7,
    "F8": F8, "F9": F9, "F10": F10, "F11": F11, "F12": F12, "F13": F13,
    "F14": F14, "F15": F15, "F16": F16, "F17": F17, "F18": F18, "F19": F19, "F20": F20
}

# Generate results and save to file
def calculate_metrics(result_dir, output_file):
    if not os.path.exists(result_dir):
        print(f"Error: The directory {result_dir} does not exist.")
        return

    metrics = {}
    for func_name, func in benchmark_functions.items():
        metrics[func_name] = {}

        for file_name in os.listdir(result_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(result_dir, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)

                for algo, results in data.items():
                    scores = [func(np.array(res.get("input", []))) for res in results]
                    metrics[func_name][algo] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "time": np.sum([res.get("time", 0) for res in results])
                    }

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure path does not include repeated 'Comparison'
    results_dir = os.path.join(base_dir, "Comparison", "Results").replace("/Comparison/Comparison", "/Comparison")
    output_file = os.path.join(results_dir, "metrics_functions_report.txt")
    calculate_metrics(results_dir, output_file)

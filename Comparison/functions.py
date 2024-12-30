import json
import pandas as pd
import numpy as np
from scipy.stats import ranksums
from os import path, makedirs

def f1_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e

def f2_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def f3_function(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def f4_function(x):
    return np.sum(np.arange(1, len(x) + 1) * x**4)

def f5_function(x):
    return np.sum(x**2)

def f6_function(x):
    return np.sum((x + 0.5)**2)

def f7_function(x):
    return -1 - np.cos(12 * np.sqrt(x[0]**2 + x[1]**2)) / (0.5 * (x[0]**2 + x[1]**2) + 2)

def f8_function(x):
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

def f9_function(x):
    return np.sum([i * np.sum(x[:i])**2 for i in range(1, len(x) + 1)])

def f10_function(x):
    return (x[0] - 1)**2 + np.sum(np.arange(2, len(x) + 1) * (2 * x[1:]**2 - x[:-1])**2)

def f11_function(x):
    return np.sum(x) + np.prod(x)

def f12_function(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6) / 3 + x[0] * x[1] + 4 * x[1]**2 - 4 * x[1]**4

def f13_function(x):
    return np.max(np.abs(x))

def f14_function(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def f15_function(x):
    return np.sum(np.sin(x) * (np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi)**20))

def f16_function(x):
    term1 = (1 + x[0] + x[1])**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)
    term2 = (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
    return term1 * term2

def f17_function(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

def f18_function(x):
    return np.sum(x**2)**2

def f19_function(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

def f20_function(x):
    return np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2)) + np.exp(-np.sum(np.sin(np.sqrt(np.abs(x)))**2))

benchmark_functions = {
    "F1": f1_function,
    "F2": f2_function,
    "F3": f3_function,
    "F4": f4_function,
    "F5": f5_function,
    "F6": f6_function,
    "F7": f7_function,
    "F8": f8_function,
    "F9": f9_function,
    "F10": f10_function,
    "F11": f11_function,
    "F12": f12_function,
    "F13": f13_function,
    "F14": f14_function,
    "F15": f15_function,
    "F16": f16_function,
    "F17": f17_function,
    "F18": f18_function,
    "F19": f19_function,
    "F20": f20_function,
}

def save_table_to_txt(data, file_path):
    with open(file_path, "w") as f:
        f.write(data)

def generate_table_1(output_dir):
    functions = [
        {"Name": "F1", "Formula": "-20 * exp(-0.2 * sqrt(sum(x^2)/d)) - exp(sum(cos(2*pi*x))/d) + 20 + e"},
        {"Name": "F2", "Formula": "10*d + sum(x^2 - 10*cos(2*pi*x))"},
        {"Name": "F3", "Formula": "sum(x^2)/4000 - prod(cos(x/sqrt(i))) + 1"},
        {"Name": "F4", "Formula": "sum(i*x^4)"},
        {"Name": "F5", "Formula": "sum(x^2)"},
        {"Name": "F6", "Formula": "sum((x + 0.5)^2)"},
        {"Name": "F7", "Formula": "-1 - cos(12 * sqrt(x1^2 + x2^2)) / (0.5*(x1^2 + x2^2) + 2)"},
        {"Name": "F8", "Formula": "0.5 + (sin(x1^2 - x2^2)^2 - 0.5) / (1 + 0.001*(x1^2 + x2^2))^2"},
        {"Name": "F9", "Formula": "sum(i * sum(x[:i])^2)"},
        {"Name": "F10", "Formula": "(x1 - 1)^2 + sum(i * (2*x[i]^2 - x[i-1])^2)"},
        {"Name": "F11", "Formula": "sum(x) + prod(x)"},
        {"Name": "F12", "Formula": "4*x1^2 - 2.1*x1^4 + (x1^6)/3 + x1*x2 + 4*x2^2 - 4*x2^4"},
        {"Name": "F13", "Formula": "max(|x|)"},
        {"Name": "F14", "Formula": "sum(|x*sin(x) + 0.1*x|)"},
        {"Name": "F15", "Formula": "sum(sin(x) * (sin(i * x^2/pi)^20))"},
        {"Name": "F16", "Formula": "complicated multi-variable function"},
        {"Name": "F17", "Formula": "another multi-variable function"},
        {"Name": "F18", "Formula": "sum(x^2)^2"},
        {"Name": "F19", "Formula": "sum(|x|)*exp(-sum(sin(x^2)))"},
        {"Name": "F20", "Formula": "sum(sin^2(x)) - exp(-sum(x^2)) + exp(-sum(sin(sqrt(|x|))^2))"}
    ]
    df = pd.DataFrame(functions)
    file_path = path.join(output_dir, "table_1_functions.txt")
    save_table_to_txt(df.to_string(index=False), file_path)
    print(f"Table 1 saved as {file_path}")

def generate_table_2(output_dir):
    parameters = [
        {"Algorithm": "SW", "Parameter": "Gap Penalty", "Value": "-1.0"},
        {"Algorithm": "SW", "Parameter": "Match Score", "Value": "2.0"},
        {"Algorithm": "SW", "Parameter": "Mismatch Penalty", "Value": "-1.0"},
        {"Algorithm": "PSO", "Parameter": "Inertia Weight (w)", "Value": "0.9"},
        {"Algorithm": "PSO", "Parameter": "Cognitive Coefficient (c1)", "Value": "1.5"},
        {"Algorithm": "PSO", "Parameter": "Social Coefficient (c2)", "Value": "1.5"},
        {"Algorithm": "ASCA-PSO", "Parameter": "Inertia Weight (w)", "Value": "0.9"},
        {"Algorithm": "ASCA-PSO", "Parameter": "Cognitive Coefficient (c1)", "Value": "1.5"},
        {"Algorithm": "ASCA-PSO", "Parameter": "Social Coefficient (c2)", "Value": "1.5"},
        {"Algorithm": "ASCA-PSO", "Parameter": "Exploration Factor (a)", "Value": "2.0"},
        {"Algorithm": "ASCA-PSO", "Parameter": "r2", "Value": "random(0, 2π)"},
        {"Algorithm": "ASCA-PSO", "Parameter": "r3", "Value": "random(0, 1)"},
        {"Algorithm": "ASCA-PSO", "Parameter": "r4", "Value": "random(0, 1)"},
        {"Algorithm": "SCA", "Parameter": "a", "Value": "2.0"},
        {"Algorithm": "SCA", "Parameter": "r1", "Value": "10.0"},
        {"Algorithm": "SCA", "Parameter": "r2", "Value": "random(0, 2π)"},
        {"Algorithm": "SCA", "Parameter": "r3", "Value": "random(0, 1)"},
        {"Algorithm": "SCA", "Parameter": "r4", "Value": "random(0, 1)"},
        {"Algorithm": "FLAT", "Parameter": "Fragment Length", "Value": "variable"},
        {"Algorithm": "FLAT", "Parameter": "Maximum Iterations", "Value": "10"}
    ]
    df = pd.DataFrame(parameters)
    file_path = path.join(output_dir, "table_2_parameters.txt")
    save_table_to_txt(df.to_string(index=False), file_path)
    print(f"Table 2 saved as {file_path}")

def generate_table(json_file, output_dir, table_name, metric_function):
    with open(json_file, "r") as f:
        data = json.load(f)

    required_input_size = {
        "F7": 2,
        "F8": 2,
        "F12": 2,
        "F16": 2,
        "F17": 2,
    }

    results = []
    for function_name, function in benchmark_functions.items():
        for algorithm, iterations in data.items():
            scores = []
            for iteration in iterations:
                if "score" in iteration:
                    numeric_values = np.array([iteration["score"]])
                else:
                    continue

                min_size = required_input_size.get(function_name, 1)
                if len(numeric_values) < min_size:
                    numeric_values = np.pad(numeric_values, (0, min_size - len(numeric_values)), constant_values=0)

                try:
                    score = function(numeric_values)
                    scores.append(score)
                except Exception as e:
                    print(f"Error computing {function_name} for {algorithm}: {e}")

            if scores:
                try:
                    result = metric_function(scores)
                    results.append({
                        "Function": function_name,
                        "Algorithm": algorithm,
                        "Metric": result
                    })
                except Exception as e:
                    print(f"Error calculating metric for {function_name}, {algorithm}: {e}")

    if results:
        df = pd.DataFrame(results)
        file_path = path.join(output_dir, f"{table_name}.txt")
        save_table_to_txt(df.to_string(index=False), file_path)
        print(f"{table_name} saved as {file_path}")
    else:
        print(f"No results generated for {table_name}")

def generate_table_6(json_file, output_dir):
    with open(json_file, "r") as f:
        data = json.load(f)

    results = []
    for function_name, function in benchmark_functions.items():
        algorithms = list(data.keys())
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo1, algo2 = algorithms[i], algorithms[j]
                scores1, scores2 = [], []

                for iteration in data[algo1]:
                    if "score" in iteration:
                        numeric_values = np.array([iteration["score"]])
                    else:
                        continue

                    try:
                        score = function(numeric_values)
                        scores1.append(score)
                    except Exception as e:
                        print(f"Error computing {function_name} for {algo1}: {e}")

                for iteration in data[algo2]:
                    if "score" in iteration:
                        numeric_values = np.array([iteration["score"]])
                    else:
                        continue

                    try:
                        score = function(numeric_values)
                        scores2.append(score)
                    except Exception as e:
                        print(f"Error computing {function_name} for {algo2}: {e}")

                if scores1 and scores2:
                    try:
                        _, p_value = ranksums(scores1, scores2)
                        results.append({
                            "Function": function_name,
                            "Algorithm 1": algo1,
                            "Algorithm 2": algo2,
                            "P-value": p_value
                        })
                    except Exception as e:
                        print(f"Error performing Wilcoxon test for {algo1} vs {algo2} on {function_name}: {e}")

    if results:
        df = pd.DataFrame(results)
        file_path = path.join(output_dir, "table_6_wilcoxon_test.txt")
        save_table_to_txt(df.to_string(index=False), file_path)
        print(f"Table 6 saved as {file_path}")
    else:
        print("No results generated for Table 6")

output_directory = "/Users/mennahtullahmabrouk/PycharmProjects/Algorthium Project/Algorithm-Project/Comparison/Results"
makedirs(output_directory, exist_ok=True)

results_file = "/Users/mennahtullahmabrouk/PycharmProjects/Algorthium Project/Algorithm-Project/Comparison/Results/combined_algorithms_results_300.json"

generate_table_1(output_directory)
generate_table_2(output_directory)
generate_table(results_file, output_directory, "table_3_average_minimum_values", np.mean)
generate_table(results_file, output_directory, "table_4_standard_deviation", np.std)
generate_table(results_file, output_directory, "table_5_elapsed_time", lambda x: np.mean(x))
generate_table_6(results_file, output_directory)

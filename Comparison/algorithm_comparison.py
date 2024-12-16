import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from fpdf import FPDF
import time
import logging
from Bio import SeqIO
from flat import flat_algorithm
from smith_waterman import smith_waterman
from particle_swarm_optimization import pso_fitness
from sine_cosine_algorithm import sine_cosine_algorithm
from sca_pso_alignment import asca_pso

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Benchmark function for algorithms
def benchmark_algorithm(algorithm_func, sequences, runs=50, fragment_length=50):
    fitness_values = []
    start_time = time.time()
    for _ in range(runs):
        if algorithm_func.__name__ == "flat_algorithm":
            score, _ = algorithm_func(sequences[0], sequences[1], fragment_length)
        elif algorithm_func.__name__ == "pso_fitness":
            from pyswarm import pso
            lb, ub = [0, 0], [len(sequences) - 1, len(sequences) - 1]
            best_params, _ = pso(algorithm_func, lb, ub, args=(sequences,), swarmsize=30, maxiter=50)
            score = smith_waterman(sequences[int(best_params[0])], sequences[int(best_params[1])])
        else:
            _, score = algorithm_func(sequences)
        fitness_values.append(score)
    elapsed_time = (time.time() - start_time) / runs
    return fitness_values, elapsed_time

# Load sequences
def load_sequences(fasta_path):
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Main comparison function
def compare_algorithms():
    # Load sequences
    fasta_path = 'Dataset/sequence.fasta'
    sequences = load_sequences(fasta_path)

    # Define algorithms with Smith-Waterman as baseline
    algorithms = {
        "Smith-Waterman (Baseline)": smith_waterman,
        "FLAT": flat_algorithm,
        "PSO": pso_fitness,
        "SCA": sine_cosine_algorithm,
        "ASCA-PSO": asca_pso
    }

    # Metrics storage
    results = {}
    logger.info("Starting benchmark for all algorithms...")
    for name, func in algorithms.items():
        logger.info(f"Running {name}...")
        scores, avg_time = benchmark_algorithm(func, sequences)
        results[name] = {
            "scores": scores,
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "avg_time": avg_time
        }

    # Generate tables
    generate_tables_and_pdf(results)

def generate_tables_and_pdf(results):
    # Table 1: Benchmark Metrics
    table1 = pd.DataFrame({
        "Algorithm": results.keys(),
        "Mean Score": [res["mean"] for res in results.values()],
        "Median Score": [res["median"] for res in results.values()],
        "Std Dev": [res["std"] for res in results.values()],
        "Avg Time (s)": [res["avg_time"] for res in results.values()]
    })

    # Table 6: P-values (comparison with Smith-Waterman baseline)
    baseline_scores = results["Smith-Waterman (Baseline)"]["scores"]
    p_values = {alg: wilcoxon(baseline_scores, res["scores"]).pvalue for alg, res in results.items() if alg != "Smith-Waterman (Baseline)"}
    table6 = pd.DataFrame({"Algorithm": list(p_values.keys()), "P-value": list(p_values.values())})

    # Save to PDF
    save_results_to_pdf(table1, table6, results)

def save_results_to_pdf(table1, table6, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Table 1: Metrics
    pdf.cell(200, 10, txt="Table 1: Benchmark Metrics", ln=True)
    for _, row in table1.iterrows():
        pdf.cell(0, 10, txt=f"{row['Algorithm']} - Mean: {row['Mean Score']:.2f}, Std Dev: {row['Std Dev']:.2f}, Time: {row['Avg Time (s)']:.4f}s", ln=True)

    # Table 6: P-values
    pdf.cell(200, 10, txt="\nTable 6: Wilcoxon Rank-Sum Test P-values", ln=True)
    for _, row in table6.iterrows():
        p_val_display = f"_{row['P-value']:.4f}_" if row['P-value'] > 0.05 else f"{row['P-value']:.4f}"
        pdf.cell(0, 10, txt=f"Baseline vs {row['Algorithm']}: P-value = {p_val_display}", ln=True)

    # Convergence Curves
    plt.figure(figsize=(10, 6))
    for alg, res in results.items():
        plt.plot(np.cumsum(res["scores"]) / np.arange(1, len(res["scores"]) + 1), label=alg)
    plt.xlabel("Runs")
    plt.ylabel("Cumulative Mean Fitness")
    plt.title("Convergence Curves")
    plt.legend()
    plt.savefig("convergence_curves.png")
    plt.close()

    pdf.add_page()
    pdf.image("convergence_curves.png", x=10, y=20, w=190)

    # Conclusions
    pdf.cell(200, 10, txt="\nConclusions:", ln=True)
    best_avg = min(results, key=lambda x: results[x]["avg_time"])
    pdf.cell(0, 10, txt=f"The fastest algorithm is {best_avg}.", ln=True)
    best_score = max(results, key=lambda x: results[x]["mean"])
    pdf.cell(0, 10, txt=f"The best performing algorithm is {best_score}.", ln=True)

    pdf.output("Comparison/Final_Results/algorithm_comparison.pdf")
    logger.info("Results saved to Comparison/Final_Results/algorithm_comparison.pdf")

if __name__ == "__main__":
    compare_algorithms()

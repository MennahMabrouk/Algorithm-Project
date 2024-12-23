import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from Comparison.algorithm_comparison import benchmark_algorithm, load_sequences
from Algorithms.flat import flat_algorithm
from Algorithms.smith_waterman import smith_waterman
from Algorithms.particle_swarm_optimization import pso_algorithm
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.asca_pso_alignment import asca_pso

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(Y-%m-%d %H:%M:%S) %(levelname)s: %(message)s')
logger = logging.getLogger()

# Load sequences with increasing lengths
def load_data():
    root_dir = Path(__file__).resolve().parent.parent
    fasta_path = root_dir / "Dataset" / "sequence.fasta"
    sequences = load_sequences(fasta_path)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")

    # Ensure sequences are sorted by length in ascending order
    sequences = sorted(sequences, key=len)
    return sequences

# Save iteration scores dynamically to text file
def save_iteration_score(filename, score_entry):
    filepath = Path("Comparison/Results") / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('a') as f:
        f.write(f"{score_entry}\n")

# Benchmark a single algorithm
def benchmark_algorithm_for_iterations(func, sequences, iteration):
    try:
        logger.info(f"Running iteration {iteration} for {func.__name__}")
        result = benchmark_algorithm(func, [sequences[0], sequences[1]], runs=1, fragment_length=20)
        score = result[0] if isinstance(result, tuple) else result
        valid_scores = [s for s in score if not np.isinf(s) and not np.isnan(s) and s > 0]

        if not valid_scores:
            raise ValueError(f"Algorithm {func.__name__} failed to produce a valid score in iteration {iteration}.")

        best_score = max(valid_scores)
        logger.info(f"Iteration {iteration} best score: {best_score}")
        return best_score
    except Exception as e:
        logger.error(f"Error in iteration {iteration} for {func.__name__}: {e}")
        return None

# Generate and save graph for scores
def generate_graph(filename, algorithms_scores, iterations):
    graph_path = Path("Comparison/Graphs") / f"{filename.replace('.txt', '.png')}"
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algorithm, scores in algorithms_scores.items():
        plt.plot(range(1, len(scores) + 1), scores, label=algorithm)

    plt.title(f'Algorithm Performance Over {iterations} Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(graph_path)
    plt.close()

    logger.info(f"Graph saved at {graph_path}")

# Run and log all algorithms for a single sequence pair
def run_algorithms_for_sequences(sequences, iterations, filename):
    algorithms = {
        "Smith-Waterman": smith_waterman,
        "FLAT": flat_algorithm,
        "PSO": pso_algorithm,
        "SCA": sine_cosine_algorithm,
        "ASCA-PSO": asca_pso
    }

    algorithms_scores = {name: [] for name in algorithms.keys()}

    for iteration in range(1, iterations + 1):
        # Dynamically select sequences for the current iteration
        if len(sequences) < 2 * iteration:
            logger.warning("Not enough sequences for the requested iteration count.")
            break

        seq_pair = sequences[(iteration - 1) * 2:(iteration - 1) * 2 + 2]

        for name, func in algorithms.items():
            logger.info(f"Running algorithm: {name} on iteration {iteration}")
            score = benchmark_algorithm_for_iterations(func, seq_pair, iteration)
            if score is not None:
                algorithms_scores[name].append(score)
                save_iteration_score(filename, f"{name}: score: {score} iteration: {iteration}")
            else:
                logger.warning(f"{name} failed to produce a valid score for iteration {iteration}.")

    generate_graph(filename, algorithms_scores, iterations)

if __name__ == "__main__":
    Path("Comparison/Results").mkdir(parents=True, exist_ok=True)

    sequences = load_data()

    # Loop for 50 iterations
    run_algorithms_for_sequences(sequences, 50, "iterations_50.txt")

    # Loop for 150 iterations
    run_algorithms_for_sequences(sequences, 150, "iterations_150.txt")

    # Loop for 250 iterations
    run_algorithms_for_sequences(sequences, 250, "iterations_250.txt")

    # Loop for 350 iterations
    run_algorithms_for_sequences(sequences, 350, "iterations_350.txt")

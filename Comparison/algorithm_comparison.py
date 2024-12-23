import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import time
import logging
from Bio import SeqIO
from pathlib import Path
from Algorithms.flat import flat_algorithm
from Algorithms.smith_waterman import smith_waterman
from Algorithms.particle_swarm_optimization import pso_algorithm
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.asca_pso_alignment import asca_pso

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# Benchmark function for algorithms
def benchmark_algorithm(algorithm_func, sequences, runs=50, fragment_length=50):
    fitness_values = []
    times = []
    visited_pairs = set()  # To track unique sequence pairs

    for i in range(runs):
        start_time = time.time()

        # Ensure unique sequence pairs
        while True:
            seq1_idx, seq2_idx = np.random.randint(0, len(sequences), size=2)
            if seq1_idx != seq2_idx and (seq1_idx, seq2_idx) not in visited_pairs:
                visited_pairs.add((seq1_idx, seq2_idx))
                break

        seq1, seq2 = sequences[seq1_idx], sequences[seq2_idx]

        try:
            if algorithm_func.__name__ == "flat_algorithm":
                result = algorithm_func(seq1, seq2, fragment_length)
                score = result[0]
            elif algorithm_func.__name__ == "pso_algorithm":
                score, _, _ = algorithm_func([seq1, seq2], num_particles=30, num_iterations=50)
            elif algorithm_func.__name__ == "smith_waterman":
                score, _, _ = smith_waterman(seq1, seq2)
            else:
                result = algorithm_func([seq1, seq2])
                score = result[1]
        except Exception as e:
            logger.error(f"Error during iteration {i + 1}: {e}")
            score = -np.inf  # Assign a bad score for failed iterations

        elapsed_time = time.time() - start_time
        fitness_values.append(score)
        times.append(elapsed_time)

        # Append results to file immediately after each iteration
        save_iteration_result(algorithm_func.__name__, i + 1, score, elapsed_time)

        if not isinstance(score, (int, float)):
            logger.error(f"Non-numeric score returned: {score}")
            raise ValueError(f"Invalid score returned by {algorithm_func.__name__} algorithm: {score}")

    return fitness_values, times


# Load sequences
def load_sequences(fasta_path):
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]


# Save scores and times for each algorithm iteration
def save_iteration_result(algorithm_name, iteration, score, elapsed_time):
    output_dir = Path("Comparison/Results/")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{algorithm_name}_results.txt"

    with open(file_path, "a") as file:  # Append mode
        file.write(f"Iteration {iteration}: Score = {score}, Time = {elapsed_time:.4f} seconds\n")
    logger.info(f"Iteration {iteration} for {algorithm_name} saved.")


# Main comparison function
def compare_algorithms():
    root_dir = Path(__file__).resolve().parent.parent
    fasta_path = root_dir / "Dataset" / "sequence.fasta"
    sequences = load_sequences(fasta_path)

    algorithms = {
        "Smith-Waterman": smith_waterman,
        "FLAT": flat_algorithm,
        "PSO": pso_algorithm,
        "SCA": sine_cosine_algorithm,
        "ASCA-PSO": asca_pso
    }

    logger.info("Starting benchmark for all algorithms...")

    # Run each algorithm independently
    for name, func in algorithms.items():
        logger.info(f"Running {name}...")
        benchmark_algorithm(func, sequences, runs=50)

    logger.info("All benchmarks completed.")


if __name__ == "__main__":
    compare_algorithms()

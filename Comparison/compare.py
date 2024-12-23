import logging
from Algorithms.flat import flat_algorithm
from Algorithms.smith_waterman import smith_waterman
from Algorithms.particle_swarm_optimization import pso_algorithm
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.asca_pso_alignment import asca_pso





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


                else:


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

import json
import logging
import os
from Algorithms.flat import flat_algorithm
from Algorithms.smith_waterman import smith_waterman
from Algorithms.particle_swarm_optimization import pso_algorithm
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.asca_pso_alignment import asca_pso

# Configure logging
logger = logging.Logger("compare_logger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_FILE = os.path.join(BASE_DIR, "protein_combinations.json")
RESULT_FILE_TEMPLATE = os.path.join(
    BASE_DIR, "Results", "combined_algorithms_results_{num_iterations}.json"
)
num_iterations = 700

# Helper functions
def load_sequence_pairs(max_pairs=None):
    if not os.path.exists(SEQUENCE_FILE):
        raise FileNotFoundError(f"{SEQUENCE_FILE} does not exist.")
    with open(SEQUENCE_FILE, "r") as f:
        data = json.load(f)

    # Expecting `data` to be a list of lists
    if not isinstance(data, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in data):
        raise ValueError("Invalid JSON format. Expected a list of pairs (each pair being a list of two sequences).")

    # Convert to a list of dictionaries for compatibility
    pairs = [{"seq1": pair[0], "seq2": pair[1]} for pair in data]

    return pairs[:max_pairs] if max_pairs else pairs


def initialize_results_file():
    result_file = RESULT_FILE_TEMPLATE.format(num_iterations=num_iterations)
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            json.dump({}, f, indent=4)
            logger.info(f"Initialized results file: {result_file}")
    return result_file


def update_json_file(result_file, algorithm_name, iteration_data):
    results = {}
    try:
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r") as f:
                results = json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {result_file}: {e}. Starting fresh.")

    if algorithm_name not in results:
        results[algorithm_name] = []

    if not isinstance(results[algorithm_name], list):
        logger.warning(f"Resetting {algorithm_name} in results as it is not a list.")
        results[algorithm_name] = []

    results[algorithm_name].append(iteration_data)

    try:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Updated {result_file} with {algorithm_name} data: {iteration_data}")
    except Exception as e:
        logger.error(f"Error writing to {result_file}: {e}")


def run_pairwise_algorithm(result_file, algorithm_name, algorithm_function, sequence_pairs, **kwargs):
    for i in range(num_iterations):
        pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
        seq1, seq2 = pairs[i]["seq1"], pairs[i]["seq2"]
        logger.info(f"Running {algorithm_name} for Iteration {i + 1}, seq1: {seq1}, seq2: {seq2}")
        try:
            score, align1, align2, time_taken = algorithm_function(seq1, seq2, **kwargs)
            iteration_data = {
                "score": score,
                "alignment": {"align1": align1, "align2": align2},
                "time": time_taken
            }
            logger.info(f"{algorithm_name} Iteration {i + 1} Data: {iteration_data}")
        except Exception as e:
            logger.error(f"Error in {algorithm_name} Iteration {i + 1}: {e}")
            iteration_data = {"score": 0, "alignment": None, "time": None}
        update_json_file(result_file, algorithm_name, iteration_data)


def run_pso(result_file, sequence_pairs, num_particles=2):
    logger.info(f"Running PSO for {num_iterations} iterations with {num_particles} pairs per iteration.")
    try:
        unique_alignments = set()
        iteration_count = 0

        while iteration_count < num_iterations:
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))

            # Filter out pairs that would result in duplicate alignments
            filtered_pairs = [
                pair for pair in pairs
                if (pair["seq1"], pair["seq2"]) not in unique_alignments and
                   (pair["seq2"], pair["seq1"]) not in unique_alignments
            ]

            if not filtered_pairs:
                logger.warning("No more unique pairs available for PSO.")
                break

            iteration_scores, _, global_best_alignment, iteration_times = pso_algorithm(
                filtered_pairs,
                num_particles=num_particles,
                num_iterations=1
            )

            for score, time_taken in zip(iteration_scores, iteration_times):
                if isinstance(global_best_alignment, tuple) and len(global_best_alignment) == 2:
                    align1, align2 = global_best_alignment
                else:
                    logger.warning("Invalid global_best_alignment structure. Skipping.")
                    continue

                # Check if alignment is unique
                if (align1, align2) in unique_alignments or (align2, align1) in unique_alignments:
                    logger.info("Duplicate alignment detected. Skipping.")
                    continue

                # Add the new alignment to the set of unique alignments
                unique_alignments.add((align1, align2))
                iteration_count += 1

                # Prepare and save iteration data
                iteration_data = {
                    "score": score,
                    "alignment": {"align1": align1, "align2": align2},
                    "time": time_taken
                }
                update_json_file(result_file, "PSO", iteration_data)

                if iteration_count >= num_iterations:
                    break

    except Exception as e:
        logger.error(f"Error in PSO: {e}")


def run_sca(result_file, sequence_pairs, num_particles=2):
    logger.info(f"Running SCA for {num_iterations} iterations with {num_particles} pairs per iteration.")
    try:
        for i in range(num_iterations):
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
            iteration_scores, global_best_pair, global_best_alignment, global_best_score, iteration_times = sine_cosine_algorithm(
                pairs,
                num_particles=num_particles,
                num_iterations=1
            )
            for score, time_taken, (align1, align2) in zip(iteration_scores, iteration_times, [global_best_alignment]):
                iteration_data = {
                    "score": score,
                    "alignment": {"align1": align1, "align2": align2},
                    "time": time_taken
                }
                update_json_file(result_file, "SCA", iteration_data)
            logger.info(f"SCA global best score: {global_best_score}, Best pair: {global_best_pair}")
    except Exception as e:
        logger.error(f"Error in SCA: {e}")


def run_asca_pso(result_file, sequence_pairs, num_particles=2):
    logger.info(f"Running ASCA-PSO for {num_iterations} iterations with {num_particles} pairs per iteration.")
    try:
        for i in range(num_iterations):
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
            iteration_scores, global_best_pair, global_best_alignment, global_best_score, iteration_times = asca_pso(
                pairs,
                num_particles=num_particles,
                num_iterations=1
            )
            for score, time_taken, (align1, align2) in zip(iteration_scores, iteration_times, [global_best_alignment]):
                iteration_data = {
                    "score": score,
                    "alignment": {"align1": align1, "align2": align2},
                    "time": time_taken
                }
                update_json_file(result_file, "ASCA-PSO", iteration_data)
            logger.info(f"ASCA-PSO global best score: {global_best_score}, Best pair: {global_best_pair}")
    except Exception as e:
        logger.error(f"Error in ASCA-PSO: {e}")


if __name__ == "__main__":
    sequence_pairs = load_sequence_pairs()
    result_file = initialize_results_file()
    run_pairwise_algorithm(result_file, "Smith-Waterman", smith_waterman, sequence_pairs)
    run_pairwise_algorithm(result_file, "FLAT", flat_algorithm, sequence_pairs, fragment_length=20)
    run_pso(result_file, sequence_pairs, num_particles=2)
    run_sca(result_file, sequence_pairs, num_particles=2)
    run_asca_pso(result_file, sequence_pairs, num_particles=2)

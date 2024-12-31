import json
import logging
import os
import numpy as np
from Algorithms.flat import flat_algorithm
from Algorithms.smith_waterman import smith_waterman
from Algorithms.particle_swarm_optimization import pso_algorithm
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.asca_pso_alignment import asca_pso

logger = logging.Logger("compare_logger")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_FILE = os.path.join(BASE_DIR, "Dataset", "protein_combinations.json")
RESULT_FILE_TEMPLATE = os.path.join(
    BASE_DIR, "Results", "combined_algorithms_results_{num_iterations}.json"
)
num_iterations = 50

def load_sequence_pairs(max_pairs=None):
    """
    Load sequence pairs from the dataset.
    """
    if not os.path.exists(SEQUENCE_FILE):
        raise FileNotFoundError(f"{SEQUENCE_FILE} does not exist.")
    with open(SEQUENCE_FILE, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in data):
        raise ValueError("Invalid JSON format. Expected a list of pairs (each pair being a list of two sequences).")
    pairs = [{"seq1": pair[0], "seq2": pair[1]} for pair in data]
    return pairs[:max_pairs] if max_pairs else pairs

def initialize_results_file():
    """
    Initialize the results JSON file if it does not already exist.
    """
    result_file = RESULT_FILE_TEMPLATE.format(num_iterations=num_iterations)
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            json.dump({}, f, indent=4)
            logger.info(f"Initialized results file: {result_file}")
    return result_file

def update_json_file(result_file, algorithm_name, iteration_data):
    """
    Update the JSON file with results from the current iteration.
    """
    try:
        # Load existing results or initialize a new structure
        if os.path.exists(result_file):
            with open(result_file, "r") as file:
                results = json.load(file)
        else:
            results = {}

        # Initialize the algorithm's result list if not present
        if algorithm_name not in results:
            results[algorithm_name] = []

        # Append the current iteration data
        results[algorithm_name].append(iteration_data)

        # Write back to the JSON file
        with open(result_file, "w") as file:
            json.dump(results, file, indent=4)
        logger.info(f"Updated {result_file} with {algorithm_name} data: {iteration_data}")
    except Exception as e:
        logger.error(f"Error writing to {result_file}: {e}")

def run_asca_pso(result_file, sequence_pairs, num_groups=3, group_size=5, asca_iterations=2):
    """
    Run the ASCA-PSO algorithm for sequence alignment.

    Args:
        result_file (str): Path to the results file.
        sequence_pairs (list): List of sequence pairs to process.
        num_groups (int): Number of groups in the ASCA-PSO algorithm.
        group_size (int): Number of agents in each group.
        asca_iterations (int): Number of iterations for ASCA-PSO per group.
    """
    logger.info(f"Running ASCA-PSO for {num_iterations} iterations with {num_groups} groups of {group_size} agents per group.")

    for i in range(num_iterations):
        try:
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
            logger.info(f"ASCA-PSO Iteration {i + 1} started with {num_groups} groups and {group_size} agents per group.")

            # Run ASCA-PSO for the current iteration
            global_best_score, global_best_alignment, total_time = asca_pso(
                pairs,
                num_groups=num_groups,
                group_size=group_size,
                num_iterations=asca_iterations
            )

            # Validate global_best_alignment
            if not global_best_alignment or len(global_best_alignment) != 2:
                raise ValueError("Invalid alignment format from ASCA-PSO.")

            align1, align2 = global_best_alignment["align1"], global_best_alignment["align2"]

            # Prepare and save iteration data
            iteration_data = {
                "score": float(global_best_score),  # Ensure it's serializable
                "alignment": {"align1": align1, "align2": align2},  # Use actual alignments
                "time": float(total_time)  # Ensure it's serializable
            }
            update_json_file(result_file, "ASCA-PSO", iteration_data)

            logger.info(f"ASCA-PSO Iteration {i + 1} Best Score: {global_best_score}, Alignment: {global_best_alignment}")

        except ValueError as ve:
            logger.error(f"Error in ASCA-PSO Iteration {i + 1}: {ve}")
            continue  # Skip this iteration if there's a value error

        except Exception as e:
            logger.error(f"Unexpected error in ASCA-PSO Iteration {i + 1}: {e}")
            continue  # Proceed to the next iteration even if one fails

    logger.info("ASCA-PSO completed all iterations successfully.")


def run_pairwise_algorithm(result_file, algorithm_name, algorithm_function, sequence_pairs, **kwargs):
    """
    Run a pairwise alignment algorithm for a specified number of iterations.
    """
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

def run_flat(result_file, sequence_pairs, fragment_length=20, flat_iterations=5):
    """
    Run the FLAT algorithm for sequence alignment with multiple iterations per sequence pair.

    Args:
        result_file (str): Path to the results file.
        sequence_pairs (list): List of sequence pairs to process.
        fragment_length (int): Length of fragments for alignment.
        flat_iterations (int): Number of FLAT optimization iterations per sequence pair.
    """
    logger.info(f"Running FLAT for {num_iterations} iterations with fragment length {fragment_length}.")
    try:
        for i in range(num_iterations):
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
            seq1, seq2 = pairs[i]["seq1"], pairs[i]["seq2"]
            logger.info(f"Running FLAT Iteration {i + 1}, seq1: {seq1}, seq2: {seq2}")
            best_score = -np.inf
            total_time = 0
            best_alignment = (None, None)
            for flat_iter in range(flat_iterations):
                try:
                    logger.debug(f"FLAT Iteration {flat_iter + 1}: seq1={type(seq1)}, seq2={type(seq2)}, fragment_length={type(fragment_length)}")
                    score, align1, align2, time_taken = flat_algorithm(seq1, seq2, fragment_length=fragment_length)
                    total_time += time_taken
                    if score > best_score:
                        best_score = score
                        best_alignment = (align1, align2)
                    logger.info(f"FLAT Iteration {flat_iter + 1}/{flat_iterations} Score: {score}")
                except Exception as e:
                    logger.error(f"Error in FLAT Iteration {flat_iter + 1}: {e}")
            iteration_data = {
                "score": best_score,
                "alignment": {"align1": best_alignment[0], "align2": best_alignment[1]},
                "time": total_time
            }
            logger.info(f"FLAT Iteration {i + 1} Best Score: {best_score}")
            update_json_file(result_file, "FLAT", iteration_data)
    except Exception as e:
        logger.error(f"Error in FLAT: {e}")


def run_pso(result_file, sequence_pairs, num_particles=2, pso_iterations=5):
    """
    Run the Particle Swarm Optimization (PSO) algorithm for sequence alignment.

    Args:
        result_file (str): Path to the results file.
        sequence_pairs (list): List of sequence pairs to process.
        num_particles (int): Number of particles (pairs) per iteration.
        pso_iterations (int): Number of PSO optimization iterations per sequence pair.
    """
    logger.info(f"Running PSO for {num_iterations} iterations with {num_particles} pairs per iteration.")
    try:
        unique_alignments = set()
        iteration_count = 0
        while iteration_count < num_iterations:
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
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
                num_iterations=pso_iterations
            )
            best_score = max(iteration_scores)
            total_time = sum(iteration_times)
            for score, time_taken in zip(iteration_scores, iteration_times):
                if isinstance(global_best_alignment, tuple) and len(global_best_alignment) == 2:
                    align1, align2 = global_best_alignment
                else:
                    logger.warning("Invalid global_best_alignment structure. Skipping.")
                    continue
                if (align1, align2) in unique_alignments or (align2, align1) in unique_alignments:
                    logger.info("Duplicate alignment detected. Skipping.")
                    continue
                unique_alignments.add((align1, align2))
                iteration_count += 1
                iteration_data = {
                    "score": best_score,
                    "alignment": {"align1": align1, "align2": align2},
                    "time": total_time
                }
                update_json_file(result_file, "PSO", iteration_data)
                if iteration_count >= num_iterations:
                    break
    except Exception as e:
        logger.error(f"Error in PSO: {e}")

def run_sca(result_file, sequence_pairs, num_particles=3, sca_iterations=5, adapt=True):
    """
    Run the Sine-Cosine Algorithm (SCA) for sequence alignment with adaptive particle adjustment.

    Args:
        result_file (str): Path to the results file.
        sequence_pairs (list): List of sequence pairs to process.
        num_particles (int): Initial number of particles (pairs) per iteration.
        sca_iterations (int): Number of SCA optimization iterations per sequence pair.
        adapt (bool): Enable adaptive adjustment of the number of particles.
    """
    logger.info(f"Running SCA for {num_iterations} iterations with {num_particles} initial particles per iteration.")
    try:
        for i in range(num_iterations):
            pairs = load_sequence_pairs(max_pairs=len(sequence_pairs))
            if len(pairs) < num_particles:
                logger.warning(f"Not enough sequence pairs for {num_particles} particles in SCA Iteration {i + 1}. "
                               f"Using {len(pairs)} pairs instead.")
                num_particles = len(pairs)
            if num_particles < 3:
                logger.warning(f"Number of particles adjusted to the minimum of 3 for SCA Iteration {i + 1}.")
                num_particles = 3

            # Prepare sequence pairs for the algorithm
            selected_pairs = pairs[:num_particles]

            logger.info(f"Running SCA Iteration {i + 1} with {len(selected_pairs)} pairs.")
            iteration_scores, global_best_pair, global_best_alignment, global_best_score, iteration_times = sine_cosine_algorithm(
                selected_pairs,
                num_particles=len(selected_pairs),
                num_iterations=sca_iterations
            )

            if adapt:
                if global_best_score > 0:
                    if global_best_score > np.mean(iteration_scores[-3:]):
                        num_particles = min(num_particles + 1, len(pairs))
                    else:  # Exploration phase
                        num_particles = max(num_particles - 1, 3)

            best_score = max(iteration_scores)
            iteration_data = {
                "score": best_score,
                "alignment": {
                    "align1": global_best_alignment[0],
                    "align2": global_best_alignment[1]
                },
                "time": sum(iteration_times)
            }
            logger.info(f"SCA Iteration {i + 1} Best Score: {best_score}")
            update_json_file(result_file, "SCA", iteration_data)
    except Exception as e:
        logger.error(f"Error in SCA: {e}")

if __name__ == "__main__":
    sequence_pairs = load_sequence_pairs()
    result_file = initialize_results_file()

    logger.info("Starting sequence alignment experiments...")
    run_pairwise_algorithm(result_file, "Smith-Waterman", smith_waterman, sequence_pairs)
    run_flat(result_file, sequence_pairs, fragment_length=20, flat_iterations=5)
    run_pso(result_file, sequence_pairs, num_particles=3, pso_iterations=5)
    run_sca(result_file, sequence_pairs, num_particles=3, sca_iterations=5)
    run_asca_pso(result_file, sequence_pairs, num_groups=3, group_size=5, asca_iterations=5)
    logger.info("All sequence alignment experiments completed successfully.")
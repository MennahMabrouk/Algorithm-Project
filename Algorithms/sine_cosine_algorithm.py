import numpy as np
import logging
import time
from Algorithms.smith_waterman import smith_waterman

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sine_cosine_algorithm(sequence_pairs, num_particles=2, num_iterations=10, a=2):
    """
    Sine-Cosine Algorithm (SCA) for sequence alignment using predefined sequence pairs.

    Args:
        sequence_pairs (list): List of sequence pairs as dictionaries {"seq1": str, "seq2": str}.
        num_particles (int): Number of particles (pairs) per iteration.
        num_iterations (int): Number of iterations for the optimization.
        a (float): Constant factor controlling exploration.

    Returns:
        iteration_scores (list): Scores for each iteration.
        global_best_pair (tuple): The best original sequences (seq1, seq2).
        global_best_alignment (tuple): Aligned sequences (aligned_seq1, aligned_seq2) for the best pair.
        global_best_score (float): The best alignment score achieved globally.
        iteration_times (list): Time taken for each iteration in seconds.
    """
    iteration_scores = []
    iteration_times = []
    total_pairs = len(sequence_pairs)

    if total_pairs < num_particles:
        raise ValueError(f"Not enough sequence pairs for {num_particles} particles.")

    global_best_score = -np.inf
    global_best_pair = (None, None)
    global_best_alignment = (None, None)

    # Initialize particles (random indices of sequence pairs)
    particle_indices = np.random.choice(range(total_pairs), num_particles, replace=False)

    for t in range(num_iterations):
        start_time = time.time()
        logger.info(f"Starting SCA iteration {t + 1}...")

        exploration_factor = a - (t / num_iterations) * a
        iteration_best_score = -np.inf

        for i in range(num_particles):
            idx = particle_indices[i]
            seq1, seq2 = sequence_pairs[idx]["seq1"], sequence_pairs[idx]["seq2"]

            r1 = exploration_factor
            r2 = np.random.uniform(0, 2 * np.pi)
            r3, r4 = np.random.uniform(0, 1, 2)

            try:
                score, aligned_seq1, aligned_seq2, _ = smith_waterman(seq1, seq2)
                if not np.isfinite(score):
                    logger.warning(f"Non-finite score detected for pair {idx}. Skipping.")
                    continue

                updated_score = score + r1 * (
                    np.sin(r2) * (r3 * global_best_score - score if global_best_score > -np.inf else 0) +
                    np.cos(r2) * (r4 * global_best_score - score if global_best_score > -np.inf else 0)
                )

                if not np.isfinite(updated_score):
                    logger.warning(f"Non-finite updated score detected for pair {idx}. Skipping.")
                    continue

                if updated_score > iteration_best_score:
                    iteration_best_score = updated_score
                if updated_score > global_best_score:
                    global_best_score = updated_score
                    global_best_pair = (seq1, seq2)
                    global_best_alignment = (aligned_seq1, aligned_seq2)

            except Exception as e:
                logger.error(f"Error during SCA iteration {t + 1} for pair {idx}: {e}")

        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)

        iteration_scores.append(iteration_best_score if iteration_best_score > -np.inf else 0)

        logger.info(f"Iteration {t + 1} completed in {iteration_time:.2f} seconds with best score: {iteration_best_score}")

        particle_indices = np.random.choice(range(total_pairs), num_particles, replace=False)

    if global_best_score == -np.inf:
        logger.warning("No valid alignment score was computed. Returning default values.")
        global_best_score = 0
        global_best_pair = (None, None)
        global_best_alignment = (None, None)

    logger.info(f"Global best score: {global_best_score}, Best pair: {global_best_pair}, Aligned sequences: {global_best_alignment}")
    return iteration_scores, global_best_pair, global_best_alignment, global_best_score, iteration_times

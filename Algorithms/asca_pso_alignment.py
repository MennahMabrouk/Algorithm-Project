import numpy as np
import logging
import time
from Algorithms.smith_waterman import smith_waterman

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def asca_pso(sequence_pairs, num_groups=5, group_size=10, num_iterations=20, w=0.9, c1=1.5, c2=1.5, a=2):
    """
    Adaptive SCA-PSO for sequence alignment using predefined sequence pairs.

    Args:
        sequence_pairs (list): List of sequence pairs as dictionaries {"seq1": str, "seq2": str}.
        num_groups (int): Number of groups in the top layer.
        group_size (int): Number of agents in each group (bottom layer).
        num_iterations (int): Number of iterations for the optimization.
        w (float): Inertia weight for PSO.
        c1, c2 (float): Cognitive and social coefficients for PSO.
        a (float): Exploration factor for SCA.

    Returns:
        score (float): The best alignment score achieved globally.
        alignment (dict): Alignment details containing "align1" and "align2".
        time (float): Total time taken for all iterations.
    """
    num_particles = num_groups
    num_agents = group_size
    total_pairs = len(sequence_pairs)

    if total_pairs < num_groups * group_size:
        raise ValueError("Insufficient sequence pairs for the specified number of groups and group size.")

    # Initialize groups and velocities
    groups = [
        np.random.choice(range(total_pairs), num_agents, replace=False) for _ in range(num_groups)
    ]
    velocities = np.zeros(num_groups)
    personal_best_scores = -np.inf * np.ones(num_groups)
    personal_best_positions = [group[0] for group in groups]  # Initial best is the first agent of each group

    global_best_score = -np.inf
    global_best_alignment = {"align1": None, "align2": None}

    total_time = 0

    for t in range(num_iterations):
        start_time = time.time()
        logger.info(f"Starting ASCA-PSO iteration {t + 1}...")

        # SCA Exploration Phase
        r1 = a - (t / num_iterations) * a
        for g, group in enumerate(groups):
            group_best_score = -np.inf
            group_best_alignment = {"align1": None, "align2": None}

            for agent_idx in group:
                seq1, seq2 = sequence_pairs[agent_idx]["seq1"], sequence_pairs[agent_idx]["seq2"]
                r2 = np.random.uniform(0, 2 * np.pi)
                r3, r4 = np.random.uniform(0, 1, 2)

                try:
                    # Run Smith-Waterman alignment
                    score, aligned_seq1, aligned_seq2, _ = smith_waterman(seq1, seq2)
                    if not np.isfinite(score):
                        continue

                    # SCA Position Update
                    if r4 < 0.5:
                        new_position = agent_idx + r1 * np.sin(r2) * abs(
                            r3 * personal_best_positions[g] - agent_idx)
                    else:
                        new_position = agent_idx + r1 * np.cos(r2) * abs(
                            r3 * personal_best_positions[g] - agent_idx)

                    new_position = int(np.clip(new_position, 0, total_pairs - 1))
                    updated_seq1, updated_seq2 = sequence_pairs[new_position]["seq1"], sequence_pairs[new_position]["seq2"]
                    updated_score, updated_aligned_seq1, updated_aligned_seq2, _ = smith_waterman(updated_seq1,
                                                                                                  updated_seq2)

                    # Update group best
                    if updated_score > group_best_score:
                        group_best_score = updated_score
                        group_best_alignment = {"align1": updated_aligned_seq1, "align2": updated_aligned_seq2}

                except Exception as e:
                    logger.error(f"Error during SCA for group {g}: {e}")

            # Update personal and global bests
            if group_best_score > personal_best_scores[g]:
                personal_best_scores[g] = group_best_score
                personal_best_positions[g] = group[0]

            if group_best_score > global_best_score:
                global_best_score = group_best_score
                global_best_alignment = group_best_alignment

        # PSO Exploitation Phase
        for i in range(num_groups):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - personal_best_positions[i])
                + c2 * r2 * (global_best_score - personal_best_scores[i])
            )
            new_position = int(np.clip(personal_best_positions[i] + velocities[i], 0, total_pairs - 1))
            personal_best_positions[i] = new_position

        # Log iteration time
        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time

        logger.info(f"Iteration {t + 1} completed in {iteration_time:.2f} seconds with best score: {global_best_score}")

    # Log and return results
    logger.info(f"Global best score: {global_best_score}, Alignment: {global_best_alignment}")
    return global_best_score, global_best_alignment, total_time

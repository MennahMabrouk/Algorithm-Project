import numpy as np
import logging
import time
from Algorithms.smith_waterman import smith_waterman

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def asca_pso(sequence_pairs, num_particles=2, num_iterations=10, w=0.9, c1=1.5, c2=1.5, a=2):
    """
    ASCA-PSO for sequence alignment using predefined sequence pairs.

    Args:
        sequence_pairs (list): List of sequence pairs as dictionaries {"seq1": str, "seq2": str}.
        num_particles (int): Number of particles (pairs) per iteration.
        num_iterations (int): Number of iterations for the optimization.
        w (float): Inertia weight for PSO.
        c1, c2 (float): Cognitive and social coefficients for PSO.
        a (float): Exploration factor for SCA.

    Returns:
        iteration_scores (list): Scores for each iteration.
        global_best_pair (tuple): The best aligned sequences (seq1, seq2).
        global_best_alignment (tuple): Aligned sequences (aligned_seq1, aligned_seq2) for the best pair.
        global_best_score (float): The best alignment score achieved globally.
        iteration_times (list): Time taken for each iteration.
    """
    iteration_scores = []
    iteration_times = []
    total_pairs = len(sequence_pairs)

    if total_pairs < num_particles:
        raise ValueError(f"Not enough sequence pairs for {num_particles} particles.")

    # Initialize particles (indices for sequence pairs)
    particles = np.random.choice(range(total_pairs), num_particles, replace=False)
    velocities = np.zeros(num_particles)  # PSO velocities
    personal_best_scores = -np.inf * np.ones(num_particles)
    personal_best_positions = particles.copy()

    global_best_score = -np.inf
    global_best_position = None
    global_best_alignment = (None, None)
    global_best_pair = (None, None)

    for t in range(num_iterations):
        start_time = time.time()
        logger.info(f"Starting ASCA-PSO iteration {t + 1}...")

        # **Bottom Layer: Sine-Cosine Algorithm (SCA)**
        r1 = a - (t / num_iterations) * a  # Exploration factor
        iteration_best_score = -np.inf

        for i, particle_idx in enumerate(particles):
            seq1, seq2 = sequence_pairs[particle_idx]["seq1"], sequence_pairs[particle_idx]["seq2"]

            # Generate random factors
            r2 = np.random.uniform(0, 2 * np.pi)
            r3, r4 = np.random.uniform(0, 1, 2)

            try:
                score, aligned_seq1, aligned_seq2, _ = smith_waterman(seq1, seq2)
                if not np.isfinite(score):
                    logger.warning(f"Non-finite score for seq1: {seq1}, seq2: {seq2}. Skipping.")
                    continue

                if r4 < 0.5:
                    updated_position = particle_idx + r1 * np.sin(r2) * abs(r3 * personal_best_positions[i] - particle_idx)
                else:
                    updated_position = particle_idx + r1 * np.cos(r2) * abs(r3 * personal_best_positions[i] - particle_idx)

                updated_position = int(np.clip(updated_position, 0, total_pairs - 1))
                seq1, seq2 = sequence_pairs[updated_position]["seq1"], sequence_pairs[updated_position]["seq2"]
                updated_score, updated_aligned_seq1, updated_aligned_seq2, _ = smith_waterman(seq1, seq2)

                if updated_score > personal_best_scores[i]:
                    personal_best_scores[i] = updated_score
                    personal_best_positions[i] = updated_position

                if updated_score > global_best_score:
                    global_best_score = updated_score
                    global_best_position = updated_position
                    global_best_pair = (seq1, seq2)
                    global_best_alignment = (updated_aligned_seq1, updated_aligned_seq2)

                if updated_score > iteration_best_score:
                    iteration_best_score = updated_score

            except Exception as e:
                logger.error(f"Error during ASCA-PSO iteration {t + 1} for particle {particle_idx}: {e}")

        # **Top Layer: Particle Swarm Optimization (PSO)**
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (personal_best_positions[i] - particles[i]) +
                c2 * r2 * (global_best_position - particles[i])
            )
            particles[i] = int(np.clip(particles[i] + velocities[i], 0, total_pairs - 1))

        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)

        logger.info(f"Iteration {t + 1} completed in {iteration_time:.2f} seconds with best score: {iteration_best_score}")
        iteration_scores.append(iteration_best_score)

    logger.info(f"Global best score: {global_best_score}, Best pair: {global_best_pair}, Aligned sequences: {global_best_alignment}")
    return iteration_scores, global_best_pair, global_best_alignment, global_best_score, iteration_times

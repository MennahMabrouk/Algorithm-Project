import numpy as np
import logging
import time
from Algorithms.smith_waterman import smith_waterman

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pso_algorithm(sequence_pairs, num_particles=2, num_iterations=10, w=0.9, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization (PSO) for sequence alignment using predefined sequence pairs.

    Args:
        sequence_pairs (list): List of sequence pairs as dictionaries {"seq1": str, "seq2": str}.
        num_particles (int): Number of particles (pairs) per iteration.
        num_iterations (int): Number of iterations.
        w (float): Inertia weight.
        c1, c2 (float): Cognitive and social coefficients.

    Returns:
        iteration_scores (list): Scores for each iteration.
        global_best_pair (tuple): The best aligned sequences.
        global_best_alignment (tuple): The alignment of the best pair.
        iteration_times (list): Time taken for each iteration.
    """
    iteration_scores = []
    iteration_times = []
    total_pairs = len(sequence_pairs)

    if total_pairs < num_particles:
        raise ValueError(f"Not enough sequence pairs for {num_particles} particles.")

    # Initialize particles
    particles = np.random.choice(range(total_pairs), num_particles, replace=False)
    velocities = np.zeros(num_particles)  # Initialize velocities
    personal_best_scores = -np.inf * np.ones(num_particles)
    personal_best_positions = particles.copy()

    global_best_score = -np.inf
    global_best_position = None
    global_best_pair = None
    global_best_alignment = None

    for t in range(num_iterations):
        logger.info(f"Starting PSO iteration {t + 1}...")
        start_time = time.time()

        iteration_best_score = -np.inf

        for i, particle_idx in enumerate(particles):
            seq1, seq2 = sequence_pairs[particle_idx]["seq1"], sequence_pairs[particle_idx]["seq2"]

            # Calculate fitness score using smith_waterman
            try:
                score, align1, align2, _ = smith_waterman(seq1, seq2)
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particle_idx

                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle_idx
                    global_best_pair = (seq1, seq2)
                    global_best_alignment = (align1, align2)

                if score > iteration_best_score:
                    iteration_best_score = score

            except Exception as e:
                logger.error(f"Error during PSO iteration {t + 1} for pair {sequence_pairs[particle_idx]}: {e}")

        # Update velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )
            # Update position
            particles[i] = int(np.clip(particles[i] + velocities[i], 0, total_pairs - 1))

        iteration_time = time.time() - start_time
        iteration_times.append(iteration_time)
        logger.info(f"Iteration {t + 1} best score: {iteration_best_score}")
        iteration_scores.append(iteration_best_score)

    logger.info(f"Global best score: {global_best_score}, Best pair: {global_best_pair}")
    return iteration_scores, global_best_pair, global_best_alignment, iteration_times

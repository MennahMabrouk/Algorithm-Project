import numpy as np
import logging
from Algorithms.smith_waterman import smith_waterman  # Import Smith-Waterman

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sine-Cosine Algorithm (SCA) for sequence alignment
def sine_cosine_algorithm(sequences, num_particles=30, num_iterations=1):
    """
    Sine-Cosine Algorithm (SCA) for sequence alignment.

    Args:
        sequences (list): List of sequences.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations.

    Returns:
        tuple: Global best particle and its score.
    """
    num_sequences = len(sequences)

    # Initialize positions and velocities for particles
    particles = np.random.randint(0, num_sequences, (num_particles, 2))  # Each particle represents a pair of sequences
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))  # Random initial velocity for particles

    # Best solutions for particles
    best_particles = particles.copy()
    best_scores = np.full(num_particles, -np.inf)  # Initialize all scores as negative infinity

    # Evaluate initial particles and update personal best scores
    for i in range(num_particles):
        seq1_idx, seq2_idx = particles[i, 0], particles[i, 1]

        # Prevent self-alignment
        if seq1_idx == seq2_idx:
            logger.warning(f"Particle {i}: Self-alignment detected, skipping evaluation.")
            continue

        # Compute Smith-Waterman score
        try:
            score, _, _ = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])
        except Exception as e:
            logger.error(f"Error evaluating Particle {i}: {e}")
            continue

        # Validate score and update
        if isinstance(score, (int, float)):
            best_scores[i] = score
        else:
            logger.warning(f"Particle {i}: Non-numeric score returned {score}")

    # Global best
    global_best_idx = np.argmax(best_scores)
    global_best = best_particles[global_best_idx]
    global_best_score = best_scores[global_best_idx]

    logger.info(f"Initial global best score = {global_best_score}, Best pair: "
                f"seq {global_best[0] + 1} and seq {global_best[1] + 1}")

    # Sine-Cosine Algorithm Main Loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update the particle position using sine and cosine functions
            r1 = np.random.rand(2)  # Random factors for sine
            r2 = np.random.rand(2)  # Random factors for cosine
            velocities[i] = velocities[i] * np.cos(r1) + r2 * np.sin(r1)
            particles[i] = np.clip(particles[i] + velocities[i], 0, num_sequences - 1).astype(int)

            # Prevent self-alignment
            if particles[i, 0] == particles[i, 1]:
                logger.warning(f"Iteration {iteration + 1}, Particle {i}: Self-alignment detected, skipping evaluation.")
                continue

            # Evaluate fitness
            seq1_idx, seq2_idx = particles[i, 0], particles[i, 1]
            try:
                score, _, _ = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])
            except Exception as e:
                logger.error(f"Error evaluating Particle {i} at Iteration {iteration + 1}: {e}")
                continue

            # Validate score and update
            if isinstance(score, (int, float)):
                if score > best_scores[i]:
                    best_scores[i] = score
                    best_particles[i] = particles[i].copy()
                if score > global_best_score:
                    global_best_score = score
                    global_best = particles[i].copy()
            else:
                logger.warning(f"Iteration {iteration + 1}, Particle {i}: Non-numeric score returned {score}")

        # Log iteration results
        logger.info(f"Iteration {iteration + 1}: Global best score = {global_best_score}, Best pair: "
                    f"seq {global_best[0] + 1} and seq {global_best[1] + 1}")

    return global_best, global_best_score

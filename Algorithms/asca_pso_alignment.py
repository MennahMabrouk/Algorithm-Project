import numpy as np
import logging
from Algorithms.smith_waterman import smith_waterman
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm  # Importing SCA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ASCA-PSO for sequence alignment
def asca_pso(sequences, num_particles=5, num_search_agents=10, num_iterations=20):
    """
    ASCA-PSO for sequence alignment using Sine-Cosine Algorithm for exploration and PSO for exploitation.

    Args:
        sequences (list): List of sequences as strings.
        num_particles (int): Number of particles in the swarm.
        num_search_agents (int): Number of search agents.
        num_iterations (int): Number of iterations for the optimization.

    Returns:
        best_alignment (tuple): The best aligned sequences.
        best_score (int): The best alignment score.
    """
    num_sequences = len(sequences)

    # Initialize positions and velocities for PSO
    x = np.zeros((num_particles, num_search_agents, 2))
    y = np.zeros((num_particles, 2))
    v = np.random.uniform(-1, 1, (num_particles, 2))

    # Track used sequence pairs to ensure uniqueness
    used_pairs = set()

    def get_unique_pair():
        """Generate a unique sequence pair."""
        available_pairs = list(set((i, j) for i in range(num_sequences) for j in range(num_sequences) if i != j) - used_pairs)
        if not available_pairs:
            raise ValueError("No more unique pairs available.")
        seq1, seq2 = available_pairs[np.random.randint(0, len(available_pairs))]
        used_pairs.add((seq1, seq2))
        return seq1, seq2

    # Initialize particles with unique sequence pairs
    logger.info("Initializing particles with unique sequence pairs...")
    for i in range(num_particles):
        for j in range(num_search_agents):
            seq1, seq2 = get_unique_pair()
            x[i, j] = [seq1, seq2]

    # Initialize personal bests (pbest) and global best (gbest)
    for i in range(num_particles):
        seq1_idx, seq2_idx = get_unique_pair()
        y[i] = [seq1_idx, seq2_idx]

    y_pbest = y.copy()

    # Compute initial global best (y_gbest)
    logger.info("Computing initial global best...")
    fitness_scores = [
        smith_waterman(
            sequences[int(y[i, 0])],
            sequences[int(y[i, 1])]
        )[0]
        for i in range(num_particles)
    ]
    global_best_index = np.argmax(fitness_scores)
    y_gbest = y[global_best_index]
    best_global_score = fitness_scores[global_best_index]
    logger.info(f"Initial global best score = {best_global_score}")

    # PSO Parameters
    w = 0.7  # inertia weight
    c1, c2 = 1.5, 1.5  # cognitive and social coefficients
    a = 2.0  # exploration factor for SCA
    stagnation_counter = 0  # Count iterations without improvement

    for t in range(num_iterations):
        logger.info(f"Starting iteration {t + 1}...")

        # Exploration: Use Sine-Cosine Algorithm (SCA) for global search
        for i in range(num_particles):
            for j in range(num_search_agents):
                r1 = np.random.uniform(0, a * (1 - t / num_iterations))
                r2 = np.random.uniform(0, 2 * np.pi)
                r3, r4 = np.random.uniform(), np.random.uniform()

                if r4 < 0.5:
                    x[i, j] += r1 * np.sin(r2) * abs(r3 * y[i] - x[i, j])
                else:
                    x[i, j] += r1 * np.cos(r2) * abs(r3 * y[i] - x[i, j])

                x[i, j] = np.clip(x[i, j], 0, num_sequences - 1)

        logger.info("Exploration phase completed. Starting exploitation phase...")
        # Exploitation: Use PSO to refine the solutions
        for i in range(num_particles):
            particle_fitness_scores = [
                smith_waterman(
                    sequences[int(x[i, j, 0])],
                    sequences[int(x[i, j, 1])]
                )[0]
                for j in range(num_search_agents)
            ]
            best_agent_index = np.argmax(particle_fitness_scores)
            best_agent = x[i, best_agent_index]

            # Update personal best
            current_best = particle_fitness_scores[best_agent_index]
            current_particle_best = smith_waterman(
                sequences[int(y[i, 0])],
                sequences[int(y[i, 1])]
            )[0]

            if current_best > current_particle_best:
                y[i] = best_agent

            # Update global best
            if current_best > best_global_score:
                y_gbest = y[i]
                best_global_score = current_best
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            # Update velocity and position
            v[i] = (
                w * v[i] +
                c1 * np.random.rand() * (y_pbest[i] - y[i]) +
                c2 * np.random.rand() * (y_gbest - y[i])
            )
            y[i] += v[i]
            y[i] = np.clip(y[i], 0, num_sequences - 1)

        # Force exploration if stagnated
        if stagnation_counter > 3:
            logger.warning(f"Stagnation detected at iteration {t + 1}. Resetting particles.")
            for i in range(num_particles):
                seq1, seq2 = get_unique_pair()
                y[i] = [seq1, seq2]
            stagnation_counter = 0

        logger.info(f"Iteration {t + 1} completed. Global best score = {best_global_score}")

    # Return best result
    seq1_idx, seq2_idx = int(y_gbest[0]), int(y_gbest[1])
    best_score = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])[0]
    logger.info(f"Final best score = {best_score}, Best pair: seq {seq1_idx + 1} and seq {seq2_idx + 1}")
    return y_gbest, best_score

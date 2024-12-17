import numpy as np
from Algorithms.smith_waterman import smith_waterman
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm  # Importing SCA

# ASCA-PSO for sequence alignment using Sine-Cosine Algorithm for exploration and PSO for exploitation
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
    x = np.random.uniform(0, num_sequences, (num_particles, num_search_agents, 2)).astype(float)
    y = np.random.uniform(0, num_sequences, (num_particles, 2)).astype(float)
    v = np.random.uniform(-1, 1, (num_particles, 2))

    # Best solutions
    y_pbest = y.copy()

    # Compute initial global best (y_gbest)
    y_gbest = y[np.argmax([
        smith_waterman(sequences[int(np.clip(np.round(y[i, 0]), 0, num_sequences - 1))],
                       sequences[int(np.clip(np.round(y[i, 1]), 0, num_sequences - 1))])
        for i in range(num_particles)
    ])]

    # Parameters
    w = 0.7  # inertia weight
    c1, c2 = 1.5, 1.5  # cognitive and social coefficients
    a = 2.0  # exploration factor for SCA

    for t in range(num_iterations):
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

                x[i, j] = np.clip(x[i, j], 0, num_sequences - 1)  # Ensure bounds

        # Exploitation: Use PSO to refine the solutions
        for i in range(num_particles):
            best_score_in_group = max([
                smith_waterman(
                    sequences[int(np.clip(np.round(x[i, j, 0]), 0, num_sequences - 1))],
                    sequences[int(np.clip(np.round(x[i, j, 1]), 0, num_sequences - 1))]
                )
                for j in range(num_search_agents)
            ])
            best_agent = x[i, np.argmax([
                smith_waterman(
                    sequences[int(np.clip(np.round(x[i, j, 0]), 0, num_sequences - 1))],
                    sequences[int(np.clip(np.round(x[i, j, 1]), 0, num_sequences - 1))]
                )
                for j in range(num_search_agents)
            ])]

            # Update particle best
            current_best = smith_waterman(
                sequences[int(np.clip(np.round(best_agent[0]), 0, num_sequences - 1))],
                sequences[int(np.clip(np.round(best_agent[1]), 0, num_sequences - 1))]
            )
            current_particle_best = smith_waterman(
                sequences[int(np.clip(np.round(y[i, 0]), 0, num_sequences - 1))],
                sequences[int(np.clip(np.round(y[i, 1]), 0, num_sequences - 1))]
            )

            if current_best > current_particle_best:
                y[i] = best_agent

            # Update global best
            if current_particle_best > smith_waterman(
                    sequences[int(np.clip(np.round(y_gbest[0]), 0, num_sequences - 1))],
                    sequences[int(np.clip(np.round(y_gbest[1]), 0, num_sequences - 1))]
            ):
                y_gbest = y[i]

            # Update velocity and position
            v[i] = (
                    w * v[i] +
                    c1 * np.random.rand() * (y_pbest[i] - y[i]) +
                    c2 * np.random.rand() * (y_gbest - y[i])
            )
            y[i] += v[i]
            y[i] = np.clip(y[i], 0, num_sequences - 1)

        # Clip y_gbest after updates
        y_gbest = np.clip(y_gbest, 0, num_sequences - 1)

    # Return best result
    seq1_idx, seq2_idx = int(np.clip(np.round(y_gbest[0]), 0, num_sequences - 1)), \
        int(np.clip(np.round(y_gbest[1]), 0, num_sequences - 1))
    best_score = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])
    return y_gbest, best_score

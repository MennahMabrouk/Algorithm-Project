import numpy as np
from Algorithms.smith_waterman import smith_waterman
import random

# Parameters for the Smith-Waterman scoring system
MATCH = 2
MISMATCH = -1
GAP = -1


def pso_fitness(params, sequences):
    """
    Fitness function for PSO to maximize Smith-Waterman alignment score.
    This function calculates the alignment score between two sequences and negates it
    to transform the maximization problem into a minimization problem for PSO.

    Args:
        params (list): List of two sequence indices.
        sequences (list): List of sequences as strings.

    Returns:
        float: Negative Smith-Waterman score since PSO minimizes by default.
    """
    seq1_idx, seq2_idx = int(params[0]), int(params[1])

    # Prevent self-alignment: If both indices are the same, return a very high value to penalize this case
    if seq1_idx == seq2_idx:
        return float('inf')  # Penalize self-alignment by assigning an infinitely bad score

    seq1 = sequences[seq1_idx]
    seq2 = sequences[seq2_idx]

    # Call smith_waterman which returns (score, aligned_seq1, aligned_seq2)
    score, _, _ = smith_waterman(seq1, seq2)
    return -score  # Negate score because PSO minimizes by default


def pso_algorithm(sequences, num_particles=30, num_iterations=50, w=0.9, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization (PSO) for finding the best sequence alignment.
    This implementation uses manual PSO equations for position and velocity updates.

    Args:
        sequences (list): List of sequences as strings.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations for the PSO algorithm.
        w (float): Inertia weight to balance exploration and exploitation.
        c1 (float): Cognitive coefficient for personal best influence.
        c2 (float): Social coefficient for global best influence.

    Returns:
        best_score (float): Best alignment score.
        best_seq1_idx (int): Index of the first sequence in the best alignment.
        best_seq2_idx (int): Index of the second sequence in the best alignment.
    """
    num_sequences = len(sequences)

    # Initialize positions (indices of sequences) and velocities
    positions = np.random.randint(0, num_sequences, (num_particles, 2))
    velocities = np.random.uniform(-1, 1, (num_particles, 2))  # Velocity range can be adjusted here

    # Best personal positions (pbest) and global best (gbest)
    pbest_positions = positions.copy()
    pbest_scores = np.array([pso_fitness(pos, sequences) for pos in positions])

    gbest_position = pbest_positions[np.argmin(pbest_scores)]  # Global best
    gbest_score = np.min(pbest_scores)  # Best score

    # Main PSO loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity and position
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)

            # Velocity update equation
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (
                        gbest_position - positions[i])

            # Random perturbation: Allow some randomness in the search space
            random_factor = np.random.uniform(0, 1)  # Random factor to introduce exploration
            positions[i] = positions[i] + velocities[i] + random_factor * np.random.randint(-5, 5, 2)

            # Ensuring positions remain within bounds
            positions[i] = np.clip(positions[i], 0, num_sequences - 1)

            # Ensure no self-alignment happens
            if positions[i][0] == positions[i][1]:
                positions[i][1] = (positions[i][
                                       1] + 1) % num_sequences  # Change seq2 to a different one if seq1 == seq2

            # Evaluate fitness (alignment score)
            score = pso_fitness(positions[i], sequences)

            # Update personal best (pbest) if the new score is better
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]

        # Update global best (gbest) if a better score is found
        min_score_idx = np.argmin(pbest_scores)
        if pbest_scores[min_score_idx] < gbest_score:
            gbest_score = pbest_scores[min_score_idx]
            gbest_position = pbest_positions[min_score_idx]

        # Print the best score and corresponding pair at each iteration
        best_seq1_idx, best_seq2_idx = int(gbest_position[0]), int(gbest_position[1])
        print(f"Iteration {iteration + 1}: Best score = {gbest_score}, "
              f"Best pair: seq {best_seq1_idx + 1} and seq {best_seq2_idx + 1}")

        # Random exploration check
        if iteration % 5 == 0:
            # Apply additional randomness to the position to enforce better exploration
            for j in range(num_particles):
                if random.random() < 0.3:  # 30% chance to reset position for further exploration
                    new_pos = np.random.randint(0, num_sequences, 2)
                    while new_pos[0] == new_pos[1]:  # Prevent self-alignment
                        new_pos = np.random.randint(0, num_sequences, 2)
                    positions[j] = new_pos
                    pbest_positions[j] = new_pos  # Reset personal best for this particle

    # Return the best result
    best_seq1_idx, best_seq2_idx = int(gbest_position[0]), int(gbest_position[1])
    return gbest_score, best_seq1_idx, best_seq2_idx

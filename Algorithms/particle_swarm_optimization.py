import numpy as np
from Algorithms.smith_waterman import smith_waterman
import random

# Parameters for Smith-Waterman scoring system
MATCH = 2
MISMATCH = -1
GAP = -1


def pso_fitness(params, sequences, visited_pairs):
    """
    Fitness function to calculate Smith-Waterman alignment score.
    Args:
        params (list): List of two sequence indices.
        sequences (list): List of sequences.
        visited_pairs (set): Set of explored pairs to avoid duplicates.

    Returns:
        float: Negative alignment score or penalty for revisited pairs.
    """
    seq1_idx, seq2_idx = int(params[0]), int(params[1])

    if seq1_idx == seq2_idx or (seq1_idx, seq2_idx) in visited_pairs:
        return float('inf')  # Penalize self-alignment or revisited pairs

    visited_pairs.add((seq1_idx, seq2_idx))
    score, _, _ = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])
    return -score  # Negate the score for PSO minimization


def pso_algorithm(sequences, num_particles=30, num_iterations=50, w=0.9, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization (PSO) for sequence alignment.
    Args:
        sequences (list): List of sequences.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations.
        w (float): Inertia weight.
        c1, c2 (float): Cognitive and social coefficients.

    Returns:
        Best alignment score and sequence pair indices.
    """
    num_sequences = len(sequences)

    # Initialize positions, velocities, and tracking
    positions = np.random.randint(0, num_sequences, (num_particles, 2))
    velocities = np.random.uniform(-1, 1, (num_particles, 2))
    visited_pairs = set()

    # Initialize personal and global bests
    pbest_positions = positions.copy()
    pbest_scores = np.array([pso_fitness(pos, sequences, visited_pairs) for pos in positions])

    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)
    last_best_score = gbest_score

    # Main PSO loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity and position
            r1, r2 = np.random.rand(2), np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (
                    gbest_position - positions[i])
            positions[i] = np.clip(positions[i] + velocities[i], 0, num_sequences - 1)

            # Prevent self-alignment
            if positions[i][0] == positions[i][1]:
                positions[i][1] = (positions[i][1] + random.randint(1, num_sequences - 1)) % num_sequences

            # Evaluate fitness
            score = pso_fitness(positions[i], sequences, visited_pairs)

            # Update personal best
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]

        # Update global best
        min_score_idx = np.argmin(pbest_scores)
        if pbest_scores[min_score_idx] < gbest_score:
            gbest_score = pbest_scores[min_score_idx]
            gbest_position = pbest_positions[min_score_idx]

        # Enforce random reset if stuck
        if gbest_score == last_best_score:
            print(f"Duplicate solution detected at Iteration {iteration + 1}. Resetting particles.")
            positions = np.random.randint(0, num_sequences, (num_particles, 2))
            velocities = np.random.uniform(-1, 1, (num_particles, 2))
            pbest_positions = positions.copy()
            pbest_scores = np.array([pso_fitness(pos, sequences, visited_pairs) for pos in positions])
        else:
            last_best_score = gbest_score

        # Print iteration result
        best_seq1_idx, best_seq2_idx = int(gbest_position[0]), int(gbest_position[1])
        print(f"Iteration {iteration + 1}: Best score = {-gbest_score}, "
              f"Best pair: seq {best_seq1_idx + 1} and seq {best_seq2_idx + 1}")

    return -gbest_score, int(gbest_position[0]), int(gbest_position[1])

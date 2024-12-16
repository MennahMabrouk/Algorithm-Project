import numpy as np
from pyswarm import pso
from Algorithms.smith_waterman import smith_waterman

def pso_fitness(params, sequences):
    """
    Fitness function for PSO to maximize Smith-Waterman alignment score.

    Args:
        params (list): List of two sequence indices as floats.
        sequences (list): List of sequences as strings.

    Returns:
        float: Negative Smith-Waterman score since PSO minimizes by default.
    """
    seq1_idx, seq2_idx = int(params[0]), int(params[1])
    seq1 = sequences[seq1_idx]
    seq2 = sequences[seq2_idx]

    # Call smith_waterman which returns (score, aligned_seq1, aligned_seq2)
    score, _, _ = smith_waterman(seq1, seq2)
    return -score  # Negate score because PSO minimizes by default


def pso_algorithm(sequences, num_particles=30, num_iterations=50):
    """
    Particle Swarm Optimization (PSO) for finding the best sequence alignment.

    Args:
        sequences (list): List of sequences as strings.
        num_particles (int): Number of particles in the swarm.
        num_iterations (int): Number of iterations for the PSO algorithm.

    Returns:
        best_score (float): Best alignment score.
        best_seq1_idx (int): Index of the first sequence in the best alignment.
        best_seq2_idx (int): Index of the second sequence in the best alignment.
    """
    num_sequences = len(sequences)
    lb = [0, 0]  # Lower bound of indices
    ub = [num_sequences - 1, num_sequences - 1]  # Upper bound of indices

    best_params, best_score = pso(
        pso_fitness, lb, ub, args=(sequences,), swarmsize=num_particles, maxiter=num_iterations
    )

    best_seq1_idx = int(best_params[0])
    best_seq2_idx = int(best_params[1])
    best_score = -best_score  # Revert negated score

    return best_score, best_seq1_idx, best_seq2_idx

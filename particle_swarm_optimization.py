import numpy as np
from Bio import SeqIO
from pyswarm import pso

# Define the scoring system
MATCH = 2  # Score for a match
MISMATCH = -1  # Score for a mismatch
GAP = -1  # Penalty for a gap


# Function to calculate the Smith-Waterman matrix
def smith_waterman(seq1, seq2):
    len_seq1, len_seq2 = len(seq1), len(seq2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = matrix[i - 1, j - 1] + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH)
            delete = matrix[i - 1, j] + GAP
            insert = matrix[i, j - 1] + GAP
            matrix[i, j] = max(0, match, delete, insert)

    max_score = np.max(matrix)
    return max_score


# PSO fitness function to maximize Smith-Waterman score
def pso_fitness(params, sequences):
    seq1_idx, seq2_idx = int(params[0]), int(params[1])  # indices of sequences
    seq1 = sequences[seq1_idx]
    seq2 = sequences[seq2_idx]

    # Calculate the Smith-Waterman score
    score = smith_waterman(seq1, seq2)
    return -score  # We negate the score since PSO minimizes by default


# Read sequences from the FASTA file
fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Setup the PSO search space
num_sequences = len(sequences)
lb = [0, 0]  # lower bound of indices
ub = [num_sequences - 1, num_sequences - 1]  # upper bound of indices

# Define the number of particles and iterations
num_particles = 30
num_iterations = 50

# Use PSO to find the best alignment
best_params, best_score = pso(pso_fitness, lb, ub, args=(sequences,), swarmsize=num_particles, maxiter=num_iterations)

# Get the best sequences
best_seq1_idx, best_seq2_idx = int(best_params[0]), int(best_params[1])
best_seq1 = sequences[best_seq1_idx]
best_seq2 = sequences[best_seq2_idx]

# Perform Smith-Waterman on the best pair
best_alignment_score = smith_waterman(best_seq1, best_seq2)

# Prepare the result for saving
result = (
    f"Best sequence pair: {best_seq1_idx + 1} (Seq {best_seq1_idx + 1}) and {best_seq2_idx + 1} (Seq {best_seq2_idx + 1})\n"
    f"Alignment score: {best_alignment_score}\n"
    f"Sequence 1: {best_seq1}\n"
    f"Sequence 2: {best_seq2}\n"
    f"\n"
    f"PSO optimization details:\n"
    f"Best parameters: {best_params}\n"
    f"Best score: {best_score}\n"
)

# Save the result to a file in the 'result' folder
output_path = 'result/pso_alignment_results.txt'
with open(output_path, 'w') as file:
    file.write(result)

print(f"PSO alignment results have been saved to {output_path}")

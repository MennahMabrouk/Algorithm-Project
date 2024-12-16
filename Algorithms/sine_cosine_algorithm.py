import numpy as np
from Bio import SeqIO

# Define the scoring system
MATCH = 2  # Score for a match
MISMATCH = -1  # Score for a mismatch
GAP = -1  # Penalty for a gap


# Function to calculate the Smith-Waterman matrix (local alignment)
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


# Sine-Cosine Algorithm (SCA) for sequence alignment
def sine_cosine_algorithm(sequences, num_particles=30, num_iterations=50):
    num_sequences = len(sequences)

    # Initialize positions and velocities for particles
    particles = np.random.randint(0, num_sequences, (num_particles, 2))  # Each particle represents a pair of sequences
    velocities = np.random.rand(num_particles, 2) * 0.1  # Random initial velocity for particles

    # Best solutions for particles
    best_particles = particles.copy()
    best_scores = np.array(
        [smith_waterman(sequences[particles[i, 0]], sequences[particles[i, 1]]) for i in range(num_particles)])

    # Global best
    global_best = best_particles[np.argmax(best_scores)]
    global_best_score = np.max(best_scores)

    # Sine-Cosine Algorithm Main Loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update the particle position using sine and cosine functions
            particles[i] = particles[i] + velocities[i]
            particles[i, 0] = int(np.clip(particles[i, 0], 0, num_sequences - 1))
            particles[i, 1] = int(np.clip(particles[i, 1], 0, num_sequences - 1))

            # Evaluate fitness
            score = smith_waterman(sequences[particles[i, 0]], sequences[particles[i, 1]])

            # Update the best solution for the particle
            if score > best_scores[i]:
                best_scores[i] = score
                best_particles[i] = particles[i].copy()

            # Update the global best solution
            if score > global_best_score:
                global_best_score = score
                global_best = particles[i].copy()

        # Update velocities using sine and cosine functions
        for i in range(num_particles):
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)
            velocities[i] = velocities[i] * np.cos(r1) + r2 * np.sin(r1)
            velocities[i] = np.clip(velocities[i], -0.1, 0.1)  # Velocity limits

    return global_best, global_best_score


# Read sequences from the FASTA file
fasta_path = 'Dataset/sequence.fasta'  # Path to your FASTA file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Run the Sine-Cosine Algorithm
best_pair, best_score = sine_cosine_algorithm(sequences)

# Get the best sequences
seq1_idx, seq2_idx = best_pair[0], best_pair[1]
best_seq1 = sequences[seq1_idx]
best_seq2 = sequences[seq2_idx]

# Perform Smith-Waterman on the best pair
alignment_score = smith_waterman(best_seq1, best_seq2)

# Prepare the result for saving
result = (
    f"Best sequence pair: {seq1_idx + 1} (Seq {seq1_idx + 1}) and {seq2_idx + 1} (Seq {seq2_idx + 1})\n"
    f"Alignment score: {alignment_score}\n"
    f"Sequence 1: {best_seq1}\n"
    f"Sequence 2: {best_seq2}\n"
    f"\n"
    f"Sine-Cosine Algorithm details:\n"
    f"Best parameters: {best_pair}\n"
    f"Best score: {best_score}\n"
)

# Save the result to a file in the 'result' folder
output_path = 'result/sca_alignment_results.txt'
with open(output_path, 'w') as file:
    file.write(result)

print(f"SCA alignment results have been saved to {output_path}")
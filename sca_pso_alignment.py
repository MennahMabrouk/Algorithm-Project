import numpy as np
from Bio import SeqIO

# Smith-Waterman scoring
MATCH = 2
MISMATCH = -1
GAP = -1

def smith_waterman(seq1, seq2):
    len_seq1, len_seq2 = len(seq1), len(seq2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = matrix[i - 1, j - 1] + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH)
            delete = matrix[i - 1, j] + GAP
            insert = matrix[i, j - 1] + GAP
            matrix[i, j] = max(0, match, delete, insert)

    return np.max(matrix)

# ASCA-PSO for sequence alignment
def asca_pso(sequences, num_particles=5, num_search_agents=10, num_iterations=20):
    num_sequences = len(sequences)

    # Initialize positions and velocities
    x = np.random.uniform(0, num_sequences, (num_particles, num_search_agents, 2))
    x = np.round(x).astype(float)
    y = np.random.uniform(0, num_sequences, (num_particles, 2))
    y = np.round(y).astype(float)
    v = np.random.uniform(-1, 1, (num_particles, 2))

    # Best solutions
    y_pbest = y.copy()
    y_gbest = y[np.argmax([smith_waterman(sequences[int(y[i, 0])], sequences[int(y[i, 1])]) for i in range(num_particles)])]

    # Parameters
    w = 0.7  # inertia weight
    c1, c2 = 1.5, 1.5  # cognitive and social coefficients
    a = 2.0  # exploration factor for SCA

    for t in range(num_iterations):
        # Update search agents (SCA layer)
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

        # Evaluate and update particles (PSO layer)
        for i in range(num_particles):
            best_score_in_group = max([
                smith_waterman(sequences[int(x[i, j, 0])], sequences[int(x[i, j, 1])])
                for j in range(num_search_agents)
            ])
            best_agent = x[i, np.argmax([
                smith_waterman(sequences[int(x[i, j, 0])], sequences[int(x[i, j, 1])])
                for j in range(num_search_agents)
            ])]

            if smith_waterman(sequences[int(best_agent[0])], sequences[int(best_agent[1])]) > \
               smith_waterman(sequences[int(y[i, 0])], sequences[int(y[i, 1])]):
                y[i] = best_agent

            if smith_waterman(sequences[int(y[i, 0])], sequences[int(y[i, 1])]) > \
               smith_waterman(sequences[int(y_gbest[0])], sequences[int(y_gbest[1])]):
                y_gbest = y[i]

            # Update velocity and position
            v[i] = w * v[i] + c1 * np.random.rand() * (y_pbest[i] - y[i]) + c2 * np.random.rand() * (y_gbest - y[i])
            y[i] += v[i]

        # Clip positions
        y = np.clip(y, 0, num_sequences - 1)

    return y_gbest, smith_waterman(sequences[int(y_gbest[0])], sequences[int(y_gbest[1])])

# Read sequences from FASTA file
fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Run ASCA-PSO
best_pair, best_score = asca_pso(sequences)

# Save results
output_path = 'result/asca_pso_alignment_results.txt'
with open(output_path, 'w') as f:
    f.write(f"Best Sequence Pair: {best_pair}\n")
    f.write(f"Best Score: {best_score}\n")

print(f"Results saved in {output_path}")

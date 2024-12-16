import numpy as np
from Bio import SeqIO
from Algorithms.smith_waterman import smith_waterman

# ASCA-PSO for sequence alignment
def asca_pso(sequences, num_particles=5, num_search_agents=10, num_iterations=20):
    num_sequences = len(sequences)

    # Initialize positions and velocities
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

                x[i, j] = np.clip(x[i, j], 0, num_sequences - 1)  # Ensure bounds

        # Evaluate and update particles (PSO layer)
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
            y[i] = np.clip(y[i], 0, num_sequences - 1)  # Ensure bounds

        # Clip y_gbest after updates
        y_gbest = np.clip(y_gbest, 0, num_sequences - 1)

    # Return best result
    seq1_idx, seq2_idx = int(np.clip(np.round(y_gbest[0]), 0, num_sequences - 1)), \
                         int(np.clip(np.round(y_gbest[1]), 0, num_sequences - 1))
    best_score = smith_waterman(sequences[seq1_idx], sequences[seq2_idx])
    return y_gbest, best_score


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

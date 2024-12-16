from Bio import SeqIO
from pso_algorithm import pso_algorithm
from smith_waterman import smith_waterman

fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

if len(sequences) < 2:
    raise ValueError("The FASTA file must contain at least two sequences.")

num_particles = 30
num_iterations = 50

best_score, best_seq1_idx, best_seq2_idx = pso_algorithm(sequences, num_particles, num_iterations)

best_seq1 = sequences[best_seq1_idx]
best_seq2 = sequences[best_seq2_idx]

output_path = 'result/pso_alignment_results.txt'
with open(output_path, 'w') as file:
    file.write(f"PSO Alignment Results:\n")
    file.write(f"Best Sequence Pair: Sequence {best_seq1_idx + 1} and Sequence {best_seq2_idx + 1}\n")
    file.write(f"Best Alignment Score: {best_score}\n")
    file.write(f"Sequence 1: {best_seq1}\n")
    file.write(f"Sequence 2: {best_seq2}\n")

print(f"PSO alignment results have been saved to '{output_path}'.")

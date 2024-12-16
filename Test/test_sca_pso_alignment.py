from pathlib import Path
from Bio import SeqIO
from Algorithms.smith_waterman import smith_waterman
from Algorithms.asca_pso_alignment import asca_pso

# Get the absolute path to the "Dataset" folder from the root of the project
root_dir = Path(__file__).resolve().parent.parent  # This points to the root directory

# Get the absolute path to the "Dataset" folder and check if it exists
fasta_path = root_dir / 'Dataset' / 'sequence.fasta'
print(f"Using fasta file at: {fasta_path}")

# Read sequences from the FASTA file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Run ASCA-PSO
best_pair, best_score = asca_pso(sequences)

# Get the best sequences
best_seq1_idx, best_seq2_idx = int(best_pair[0]), int(best_pair[1])
best_seq1 = sequences[best_seq1_idx]
best_seq2 = sequences[best_seq2_idx]

# Perform Smith-Waterman to get part of the alignment
score, aligned_seq1, aligned_seq2 = smith_waterman(best_seq1, best_seq2)

# Print the results
print(f"Best Sequence Pair: Sequence {best_seq1_idx + 1} and Sequence {best_seq2_idx + 1}")
print(f"Best Alignment Score: {best_score}")
print(f"Best Sequence 1: {best_seq1}")
print(f"Best Sequence 2: {best_seq2}")
print(f"Alignment Score: {score}")
print(f"Part of the Alignment:\n{aligned_seq1}\n{aligned_seq2}")

# Define the result path
result_path = root_dir / 'Result'
result_path.mkdir(parents=True, exist_ok=True)  # Create the Result folder if it doesn't exist
output_path = result_path / 'asca_pso_alignment_results.txt'

# Save the results to a file
with open(output_path, 'w') as file:
    file.write(f"Best Sequence Pair: Sequence {best_seq1_idx + 1} and Sequence {best_seq2_idx + 1}\n")
    file.write(f"Best Alignment Score: {best_score}\n")
    file.write(f"Best Sequence 1: {best_seq1}\n")
    file.write(f"Best Sequence 2: {best_seq2}\n")
    file.write(f"Alignment Score: {score}\n")
    file.write(f"Part of the Alignment:\n{aligned_seq1}\n{aligned_seq2}\n")

print(f"Results saved to {output_path}")

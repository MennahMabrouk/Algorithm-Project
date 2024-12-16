from Bio import SeqIO
from pathlib import Path
import logging
from Algorithms.flat import flat_algorithm

# Correct way to join paths using Path from pathlib
fasta_path = Path('Dataset').joinpath("sequence.fasta")

# Read sequences from the FASTA file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Check if there are at least 2 sequences
if len(sequences) < 2:
    raise ValueError("The FASTA file must contain at least two sequences.")

# Parameters for alignment
fragment_length = 50
max_iterations = 20

# Run FLAT algorithm on the first two sequences
sequence1, sequence2 = sequences[0], sequences[1]
score, fragments = flat_algorithm(sequence1, sequence2, fragment_length, max_iterations)

# Save results
output_path = Path('Result').joinpath("flat_results.txt")
with open(output_path, 'w') as f:
    f.write(f"FLAT Alignment Results:\n")
    f.write(f"Best Score: {score}\n")
    if fragments:
        f.write(f"Fragments aligned: Sequence1[{fragments[0]}:{fragments[0] + fragment_length}] and "
                f"Sequence2[{fragments[1]}:{fragments[1] + fragment_length}]\n")

print(f"Results saved in {output_path}")

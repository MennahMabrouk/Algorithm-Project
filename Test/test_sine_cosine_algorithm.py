import numpy as np
from Bio import SeqIO
from Algorithms.sine_cosine_algorithm import sine_cosine_algorithm
from Algorithms.smith_waterman import smith_waterman

from pathlib import Path

# Set the root directory for dynamic path handling
root_dir = Path(__file__).resolve().parent.parent  # Root directory of the project

# Get the absolute path to the "Dataset" folder from the root of the project
fasta_path = root_dir / 'Dataset' / 'sequence.fasta'

# Check if the file exists
if not fasta_path.exists():
    raise FileNotFoundError(f"The file {fasta_path} was not found.")

print(f"Using fasta file at: {fasta_path}")

# Read sequences from the FASTA file
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

# Get the absolute path to the "Result" folder from the root of the project
result_path = root_dir / 'Result'

# Ensure the result directory exists
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Define the output path for the results
output_path = result_path / 'sca_alignment_results.txt'

# Save the result to a file in the 'Result' folder
with open(output_path, 'w') as file:
    file.write(result)

print(f"SCA alignment results have been saved to {output_path}")

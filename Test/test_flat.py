import sys
from pathlib import Path

# Add the root directory to the system path so Python can find the Algorithms module
root_dir = Path(__file__).resolve().parent.parent  # This points to the root directory
sys.path.append(str(root_dir))

from Bio import SeqIO
from Algorithms.flat import flat_algorithm  # Now this should work

# Get the absolute path to the "Dataset" folder from the root of the project
fasta_path = root_dir / 'Dataset' / 'sequence.fasta'

# Check if the file exists
if not fasta_path.exists():
    raise FileNotFoundError(f"The file {fasta_path} was not found.")

print(f"Using fasta file at: {fasta_path}")

# Read sequences from the FASTA file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

if len(sequences) < 2:
    raise ValueError("The FASTA file must contain at least two sequences.")

fragment_length = 50
max_iterations = 20

sequence1, sequence2 = sequences[0], sequences[1]
score, fragments, aligned_seq1, aligned_seq2 = flat_algorithm(sequence1, sequence2, fragment_length, max_iterations)

# Get the absolute path to the "Result" folder from the root of the project
result_path = root_dir / 'Result'

# Check if the Result directory exists
if not result_path.exists():
    raise FileNotFoundError(f"The directory {result_path} was not found.")

output_path = result_path / "flat_results.txt"

# Writing the results to the file
with open(output_path, 'w') as f:
    f.write(f"FLAT Alignment Results:\n")
    f.write(f"Best Score: {score}\n")
    if fragments:
        f.write(f"Fragments aligned: Sequence1[{fragments[0]}:{fragments[0] + fragment_length}] and "
                f"Sequence2[{fragments[1]}:{fragments[1] + fragment_length}]\n")
    f.write(f"\nBest Alignment Strand:\n")
    f.write(f"Sequence 1: {aligned_seq1}\n")
    f.write(f"Sequence 2: {aligned_seq2}\n")

print(f"Results saved in {output_path}")

# Print the best sequences and their alignment
print(f"\nBest Alignment Details:")
print(f"Best Score: {score}")
print(f"Best Sequence Pair: Sequence 1 and Sequence 2")
print(f"Aligned Sequences:\nSequence 1: {aligned_seq1}\nSequence 2: {aligned_seq2}")

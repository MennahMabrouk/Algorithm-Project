import sys
from pathlib import Path
from Bio import SeqIO
from Algorithms.particle_swarm_optimization import pso_algorithm


# Add the root directory to the system path so Python can find the Algorithms module
root_dir = Path(__file__).resolve().parent.parent  # This points to the root directory of the project
sys.path.append(str(root_dir))  # Add the root directory to sys.path

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

# Define parameters for ASCA-PSO
num_particles = 30
num_iterations = 50

# Run ASCA-PSO to find the best sequence pair
best_score, best_seq1_idx, best_seq2_idx = pso_algorithm(sequences, num_particles, num_iterations)

# Get the best sequences
best_seq1 = sequences[best_seq1_idx]
best_seq2 = sequences[best_seq2_idx]

# Get the absolute path to the "Result" folder from the root of the project
result_path = root_dir / 'Result'

# Check if the Result directory exists
if not result_path.exists():
    raise FileNotFoundError(f"The directory {result_path} was not found.")

# Define the output path for the results
output_path = result_path / 'asca_pso_alignment_results.txt'

# Writing the results to the file
with open(output_path, 'w') as file:
    file.write(f"ASCA-PSO Alignment Results:\n")
    file.write(f"Best Sequence Pair: Sequence {best_seq1_idx + 1} and Sequence {best_seq2_idx + 1}\n")
    file.write(f"Best Alignment Score: {best_score}\n")
    file.write(f"Sequence 1: {best_seq1}\n")
    file.write(f"Sequence 2: {best_seq2}\n")

print(f"ASCA-PSO alignment results have been saved to '{output_path}'.")

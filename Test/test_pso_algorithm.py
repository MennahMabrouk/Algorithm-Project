import sys
from pathlib import Path
from Bio import SeqIO
from pyswarm import pso  # Ensure that pso is imported
from Algorithms.smith_waterman import smith_waterman

# Add the root directory to the system path so Python can find the Algorithms module
root_dir = Path(__file__).resolve().parent.parent  # This points to the root directory
sys.path.append(str(root_dir))

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

# Define parameters for PSO
num_particles = 30
num_iterations = 50

# Function to modify the pso_algorithm to ensure it doesn't compare the same sequence
def pso_algorithm_no_self_alignment(sequences, num_particles=30, num_iterations=50):
    """
    Modified PSO to avoid aligning the same sequence with itself.
    """
    num_sequences = len(sequences)
    lb = [0, 0]  # Lower bound of indices
    ub = [num_sequences - 1, num_sequences - 1]  # Upper bound of indices

    best_params, best_score = pso(
        pso_fitness_no_self, lb, ub, args=(sequences,), swarmsize=num_particles, maxiter=num_iterations
    )

    best_seq1_idx = int(best_params[0])
    best_seq2_idx = int(best_params[1])
    best_score = -best_score  # Revert negated score

    return best_score, best_seq1_idx, best_seq2_idx

# Modified fitness function to ensure no self-alignment
def pso_fitness_no_self(params, sequences):
    """
    Fitness function for PSO to maximize Smith-Waterman alignment score, ensuring no self-alignment.
    """
    seq1_idx, seq2_idx = int(params[0]), int(params[1])

    # Ensure no self-alignment
    if seq1_idx == seq2_idx:
        return float('inf')  # Assign a very high score if aligning the same sequence

    seq1 = sequences[seq1_idx]
    seq2 = sequences[seq2_idx]

    # Call smith_waterman which returns (score, aligned_seq1, aligned_seq2)
    score, _, _ = smith_waterman(seq1, seq2)
    return -score  # Negate score because PSO minimizes by default

# Run PSO to find the best sequence pair
best_score, best_seq1_idx, best_seq2_idx = pso_algorithm_no_self_alignment(sequences, num_particles, num_iterations)

# Get the best sequences
best_seq1 = sequences[best_seq1_idx]
best_seq2 = sequences[best_seq2_idx]

# Get the absolute path to the "Result" folder from the root of the project
result_path = root_dir / 'Result'

# Check if the Result directory exists
if not result_path.exists():
    raise FileNotFoundError(f"The directory {result_path} was not found.")

# Define the output path for the results
output_path = result_path / 'pso_alignment_results.txt'

# Writing the results to the file
with open(output_path, 'w') as file:
    file.write(f"PSO Alignment Results:\n")
    file.write(f"Best Sequence Pair: Sequence {best_seq1_idx + 1} and Sequence {best_seq2_idx + 1}\n")
    file.write(f"Best Alignment Score: {best_score}\n")
    file.write(f"Sequence 1: {best_seq1}\n")
    file.write(f"Sequence 2: {best_seq2}\n")

print(f"PSO alignment results have been saved to '{output_path}'.")
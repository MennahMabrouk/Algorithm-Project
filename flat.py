import numpy as np
from Bio import SeqIO

# Smith-Waterman scoring parameters
MATCH = 2
MISMATCH = -1
GAP = -1


def smith_waterman(seq1, seq2):
    """Smith-Waterman algorithm to find the LCCS between two sequences."""
    len_seq1, len_seq2 = len(seq1), len(seq2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = matrix[i - 1, j - 1] + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH)
            delete = matrix[i - 1, j] + GAP
            insert = matrix[i, j - 1] + GAP
            matrix[i, j] = max(0, match, delete, insert)

    return np.max(matrix)


def flat_algorithm(sequence1, sequence2, fragment_length):
    """
    Fragmented Local Aligner Technique (FLAT) to detect LCCS.

    Parameters:
        sequence1: First sequence as a string.
        sequence2: Second sequence as a string.
        fragment_length: Length of fragments for alignment.

    Returns:
        LCCS score and positions of alignment.
    """
    len_seq1, len_seq2 = len(sequence1), len(sequence2)
    best_score = 0
    best_fragments = None

    # Iterate over all possible fragments in sequence1
    for i in range(0, len_seq1 - fragment_length + 1):
        fragment1 = sequence1[i:i + fragment_length]

        # Iterate over all possible fragments in sequence2
        for j in range(0, len_seq2 - fragment_length + 1):
            fragment2 = sequence2[j:j + fragment_length]

            # Compute Smith-Waterman score for the fragments
            score = smith_waterman(fragment1, fragment2)
            if score > best_score:
                best_score = score
                best_fragments = (i, j)

    return best_score, best_fragments


# Read sequences from a FASTA file
fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Ensure at least two sequences exist in the FASTA file
if len(sequences) < 2:
    raise ValueError("The FASTA file must contain at least two sequences.")

# Define fragment length for alignment
fragment_length = 50  # Adjust based on your requirements

# Run FLAT on the first two sequences in the FASTA file
sequence1, sequence2 = sequences[0], sequences[1]
score, fragments = flat_algorithm(sequence1, sequence2, fragment_length)

# Save results
output_path = 'result/flat_results.txt'
with open(output_path, 'w') as f:
    f.write(f"FLAT Alignment Results:\n")
    f.write(f"Best Score: {score}\n")
    if fragments:
        f.write(f"Fragments aligned: Sequence1[{fragments[0]}:{fragments[0] + fragment_length}] and "
                f"Sequence2[{fragments[1]}:{fragments[1] + fragment_length}]\n")

print(f"Results saved in {output_path}")

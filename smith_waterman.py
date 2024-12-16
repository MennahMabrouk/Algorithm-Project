import numpy as np
from Bio import SeqIO

# Define the scoring system
MATCH = 2  # Score for a match
MISMATCH = -1  # Score for a mismatch
GAP = -1  # Penalty for a gap


# Function to calculate the Smith-Waterman matrix
def smith_waterman(seq1, seq2):
    # Create a matrix with dimensions (len(seq1)+1) x (len(seq2)+1)
    len_seq1, len_seq2 = len(seq1), len(seq2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Fill the matrix
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = matrix[i - 1, j - 1] + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH)
            delete = matrix[i - 1, j] + GAP
            insert = matrix[i, j - 1] + GAP
            matrix[i, j] = max(0, match, delete, insert)

    # Find the maximum score in the matrix
    max_score = np.max(matrix)

    # Backtrack to find the optimal local alignment
    align1, align2 = "", ""
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)

    while matrix[i, j] > 0:
        current_score = matrix[i, j]
        diagonal_score = matrix[i - 1, j - 1]
        up_score = matrix[i - 1, j]
        left_score = matrix[i, j - 1]

        if current_score == diagonal_score + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH):
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif current_score == up_score + GAP:
            align1 = seq1[i - 1] + align1
            align2 = "-" + align2
            i -= 1
        else:  # current_score == left_score + GAP
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    return max_score, align1, align2


# Read sequences from the FASTA file
fasta_path = 'Dataset/sequence.fasta'
sequences = []

# Parse the FASTA file
for record in SeqIO.parse(fasta_path, "fasta"):
    sequences.append(str(record.seq))

# Open a file to save the output
output_file = 'sw_alignment_results.txt'

with open(output_file, 'w') as f:
    # Perform pairwise alignment for all sequences (just an example with first 3)
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = sequences[i]
            seq2 = sequences[j]

            # Perform Smith-Waterman alignment
            score, aligned_seq1, aligned_seq2 = smith_waterman(seq1, seq2)

            # Write the results to the file
            f.write(f"Alignment between sequence {i + 1} and sequence {j + 1}:\n")
            f.write(f"Max score: {score}\n")
            f.write(f"Aligned sequences:\n")
            f.write(f"{aligned_seq1}\n")
            f.write(f"{aligned_seq2}\n")
            f.write("-" * 50 + "\n")

print("Alignment results have been saved to 'sw_alignment_results.txt'")

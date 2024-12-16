import numpy as np

# Define the scoring system
MATCH = 2  # Score for a match
MISMATCH = -1  # Score for a mismatch
GAP = -1  # Penalty for a gap

def smith_waterman(seq1, seq2):
    """
    Perform local alignment of two sequences using the Smith-Waterman algorithm.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.

    Returns:
        max_score (int): The maximum alignment score.
        align1 (str): The aligned representation of seq1.
        align2 (str): The aligned representation of seq2.
    """
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

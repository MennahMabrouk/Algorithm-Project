import numpy as np
import time

MATCH = 2
MISMATCH = -1
GAP = -1

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
        execution_time (float): Time taken to compute the alignment, in seconds.
    """
    start_time = time.time()  # Start the timer

    len_seq1, len_seq2 = len(seq1), len(seq2)
    matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Fill the scoring matrix
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            match = matrix[i - 1, j - 1] + (MATCH if seq1[i - 1] == seq2[j - 1] else MISMATCH)
            delete = matrix[i - 1, j] + GAP
            insert = matrix[i, j - 1] + GAP
            matrix[i, j] = max(0, match, delete, insert)

    max_score = np.max(matrix)

    # Traceback to construct alignments
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
        else:
            align1 = "-" + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    end_time = time.time()  # End the timer
    execution_time = end_time - start_time

    return max_score, align1, align2, execution_time

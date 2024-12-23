import logging
import numpy as np
import random
import time
from Algorithms.smith_waterman import smith_waterman

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def flat_algorithm(sequence1, sequence2, fragment_length, max_iterations=10):
    """
    Fragmented Local Aligner Technique (FLAT) with exploration and exploitation.

    Parameters:
        sequence1: First sequence as a string.
        sequence2: Second sequence as a string.
        fragment_length: Length of fragments for alignment.
        max_iterations: Number of iterations for exploration and exploitation.

    Returns:
        best_score (int): The highest alignment score found.
        aligned_seq1 (str): The aligned representation of sequence1.
        aligned_seq2 (str): The aligned representation of sequence2.
        execution_time (float): Total execution time across iterations.
    """
    len_seq1, len_seq2 = len(sequence1), len(sequence2)
    if len_seq1 < fragment_length or len_seq2 < fragment_length:
        raise ValueError(f"Fragment length {fragment_length} exceeds sequence lengths.")

    best_score = -np.inf
    aligned_seq1, aligned_seq2 = "", ""
    total_execution_time = 0

    def random_exploration_method(len_seq1, len_seq2, method):
        if method == 1:
            p1 = random.randint(0, len_seq1 - fragment_length)
            p2 = random.randint(0, len_seq2 - fragment_length)
        elif method == 2:
            region = random.choice(['start', 'mid', 'end'])
            if region == 'start' and len_seq1 // 3 >= fragment_length and len_seq2 // 3 >= fragment_length:
                p1 = random.randint(0, len_seq1 // 3 - fragment_length)
                p2 = random.randint(0, len_seq2 // 3 - fragment_length)
            elif region == 'mid' and 2 * len_seq1 // 3 >= fragment_length and 2 * len_seq2 // 3 >= fragment_length:
                p1 = random.randint(len_seq1 // 3, 2 * len_seq1 // 3 - fragment_length)
                p2 = random.randint(len_seq2 // 3, 2 * len_seq2 // 3 - fragment_length)
            elif region == 'end' and len_seq1 >= 2 * len_seq1 // 3 + fragment_length and len_seq2 >= 2 * len_seq2 // 3 + fragment_length:
                p1 = random.randint(2 * len_seq1 // 3, len_seq1 - fragment_length)
                p2 = random.randint(2 * len_seq2 // 3, len_seq2 - fragment_length)
            else:
                p1 = random.randint(0, len_seq1 - fragment_length)
                p2 = random.randint(0, len_seq2 - fragment_length)
        else:
            interval = fragment_length * random.randint(1, 3)
            valid_range1 = range(0, len_seq1 - fragment_length, interval)
            valid_range2 = range(0, len_seq2 - fragment_length, interval)
            if valid_range1 and valid_range2:
                p1 = random.choice(valid_range1)
                p2 = random.choice(valid_range2)
            else:
                p1 = random.randint(0, len_seq1 - fragment_length)
                p2 = random.randint(0, len_seq2 - fragment_length)

        return p1, p2

    for iteration in range(max_iterations):
        start_time = time.time()
        try:
            exploration_method = random.choice([1, 2, 3])
            p1, p2 = random_exploration_method(len_seq1, len_seq2, exploration_method)
            fragment1 = sequence1[p1:p1 + fragment_length]
            fragment2 = sequence2[p2:p2 + fragment_length]
            score, aligned_fragment1, aligned_fragment2, _ = smith_waterman(fragment1, fragment2)

            if not np.isfinite(score):
                logger.warning(f"Non-finite score detected for iteration {iteration + 1}. Skipping.")
                continue

            if score > best_score:
                best_score = score
                aligned_seq1, aligned_seq2 = aligned_fragment1, aligned_fragment2
        except ValueError as e:
            logger.error(f"Error during fragment selection in iteration {iteration + 1}: {e}")
        finally:
            end_time = time.time()
            total_execution_time += end_time - start_time

    if best_score == -np.inf:
        logger.warning("No valid alignment score was computed. Returning default values.")
        best_score = 0
        aligned_seq1, aligned_seq2 = "", ""

    return best_score, aligned_seq1, aligned_seq2, total_execution_time

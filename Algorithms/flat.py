import numpy as np
from Algorithms.smith_waterman import smith_waterman
import random

def flat_algorithm(sequence1, sequence2, fragment_length, max_iterations=10):
    """
    Fragmented Local Aligner Technique (FLAT) with exploration and exploitation.

    Parameters:
        sequence1: First sequence as a string.
        sequence2: Second sequence as a string.
        fragment_length: Length of fragments for alignment.
        max_iterations: Number of iterations for exploration and exploitation.

    Returns:
        best_score (int): The highest LCCS score found.
        best_fragments (tuple): Start positions of the best aligned fragments.
        aligned_seq1 (str): Aligned sequence1.
        aligned_seq2 (str): Aligned sequence2.
    """

    def random_exploration_method(len_seq1, len_seq2, method):
        """Random exploration strategies to select fragment positions."""
        if method == 1:  # Uniform random selection
            p1 = random.randint(0, len_seq1 - fragment_length)
            p2 = random.randint(0, len_seq2 - fragment_length)
        elif method == 2:  # Biased random selection towards start/mid/end
            region = random.choice(['start', 'mid', 'end'])
            if region == 'start':
                p1 = random.randint(0, len_seq1 // 3 - fragment_length)
                p2 = random.randint(0, len_seq2 // 3 - fragment_length)
            elif region == 'mid':
                p1 = random.randint(len_seq1 // 3, 2 * len_seq1 // 3 - fragment_length)
                p2 = random.randint(len_seq2 // 3, 2 * len_seq2 // 3 - fragment_length)
            else:
                p1 = random.randint(2 * len_seq1 // 3, len_seq1 - fragment_length)
                p2 = random.randint(2 * len_seq2 // 3, len_seq2 - fragment_length)
        else:  # Interval-based random selection
            interval = fragment_length * random.randint(1, 3)
            p1 = random.choice(range(0, len_seq1 - fragment_length, interval))
            p2 = random.choice(range(0, len_seq2 - fragment_length, interval))
        return p1, p2

    len_seq1, len_seq2 = len(sequence1), len(sequence2)
    best_score = 0
    best_fragments = None
    aligned_seq1, aligned_seq2 = "", ""

    for _ in range(max_iterations):
        # Randomly select exploration strategy
        exploration_method = random.choice([1, 2, 3])

        # Exploration Phase: Select fragments using chosen strategy
        p1, p2 = random_exploration_method(len_seq1, len_seq2, exploration_method)
        fragment1 = sequence1[p1:p1 + fragment_length]
        fragment2 = sequence2[p2:p2 + fragment_length]

        # Exploitation Phase: Evaluate Smith-Waterman score
        score, aligned_fragment1, aligned_fragment2 = smith_waterman(fragment1, fragment2)
        if score > best_score:
            best_score = score
            best_fragments = (p1, p2)
            aligned_seq1, aligned_seq2 = aligned_fragment1, aligned_fragment2

    return best_score, best_fragments, aligned_seq1, aligned_seq2

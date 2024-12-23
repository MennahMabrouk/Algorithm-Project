import logging
import numpy as np
from Algorithms.smith_waterman import smith_waterman
import random

# Set up logging
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
        best_score (int): The highest LCCS score found.
        best_fragments (tuple): Start positions of the best aligned fragments.
        aligned_seq1 (str): Aligned sequence1.
        aligned_seq2 (str): Aligned sequence2.
    """

    def random_exploration_method(len_seq1, len_seq2, method):
        """Random exploration strategies to select fragment positions."""
        if len_seq1 < fragment_length or len_seq2 < fragment_length:
            logger.warning(
                f"Fragment length {fragment_length} exceeds sequence lengths. Reducing fragment length to fit."
            )
            return 0, 0  # Fallback to starting positions

        if method == 1:  # Uniform random selection
            p1 = random.randint(0, len_seq1 - fragment_length)
            p2 = random.randint(0, len_seq2 - fragment_length)
            logger.debug(f"Uniform selection: p1={p1}, p2={p2}")
        elif method == 2:  # Biased random selection towards start/mid/end
            region = random.choice(['start', 'mid', 'end'])
            logger.debug(f"Selected region for biased selection: {region}")
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
                logger.warning(f"Invalid region selection for {region}. Falling back to uniform selection.")
                p1 = random.randint(0, len_seq1 - fragment_length)
                p2 = random.randint(0, len_seq2 - fragment_length)
        else:  # Interval-based random selection
            interval = fragment_length * random.randint(1, 3)
            valid_range1 = range(0, len_seq1 - fragment_length, interval)
            valid_range2 = range(0, len_seq2 - fragment_length, interval)
            if not valid_range1 or not valid_range2:
                logger.warning("Invalid interval-based range; falling back to uniform random selection.")
                p1 = random.randint(0, len_seq1 - fragment_length)
                p2 = random.randint(0, len_seq2 - fragment_length)
            else:
                p1 = random.choice(valid_range1)
                p2 = random.choice(valid_range2)

        return p1, p2

    # Validate input types
    if not isinstance(sequence1, str) or not isinstance(sequence2, str):
        raise ValueError("Both sequence1 and sequence2 must be strings.")
    if not isinstance(fragment_length, int) or not isinstance(max_iterations, int):
        raise ValueError("fragment_length and max_iterations must be integers.")

    len_seq1, len_seq2 = len(sequence1), len(sequence2)
    best_score = 0
    best_fragments = None
    aligned_seq1, aligned_seq2 = "", ""

    for iteration in range(max_iterations):
        # Randomly select exploration strategy
        exploration_method = random.choice([1, 2, 3])
        logger.debug(f"Iteration {iteration + 1}: Using exploration method {exploration_method}")

        # Exploration Phase: Select fragments using chosen strategy
        try:
            p1, p2 = random_exploration_method(len_seq1, len_seq2, exploration_method)
        except ValueError as e:
            logger.error(f"Error during fragment selection: {e}")
            continue  # Skip to next iteration

        fragment1 = sequence1[p1:p1 + fragment_length]
        fragment2 = sequence2[p2:p2 + fragment_length]

        # Exploitation Phase: Evaluate Smith-Waterman score
        score, aligned_fragment1, aligned_fragment2 = smith_waterman(fragment1, fragment2)
        logger.debug(f"Score for fragments {p1}-{p1 + fragment_length} and {p2}-{p2 + fragment_length}: {score}")

        if score > best_score:
            best_score = score
            best_fragments = (p1, p2)
            aligned_seq1, aligned_seq2 = aligned_fragment1, aligned_fragment2
            logger.info(f"New best score: {best_score} with fragments {best_fragments}")

    return best_score, best_fragments, aligned_seq1, aligned_seq2

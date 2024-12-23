import json
import logging
from itertools import combinations
from pathlib import Path
from Bio import SeqIO


def load_data():
    """
    Load sequences from the dataset and sort them by length.
    """
    root_dir = Path(__file__).resolve().parent.parent
    fasta_path = root_dir / "Dataset" / "sequence.fasta"
    sequences = []
    try:
        with open(fasta_path, "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                sequences.append(str(record.seq))
        logging.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    except Exception as e:
        logging.error(f"Error loading sequences from {fasta_path}: {e}")
        return []
    return sorted(sequences, key=len)


def generate_combinations(sequences):
    """
    Generate all possible combinations of sequence pairs from a list of sequences.
    Combinations are sorted by the average length of the two sequences and saved to a file.
    """
    try:
        if len(sequences) != 100:
            logging.warning(
                f"Expected 100 sequences, but {len(sequences)} were loaded. Adjust the dataset to ensure 100 sequences."
            )
            return

        pairs = combinations(sequences, 2)
        sorted_pairs = sorted(pairs, key=lambda x: (len(x[0]) + len(x[1])) / 2)

        root_dir = Path(__file__).resolve().parent
        output_file = root_dir / "Results" / "sequence_combinations.json"

        if not output_file.parent.exists():
            logging.error(f"Required directory does not exist: {output_file.parent}")
            return

        sequence_data = {
            "combinations": [
                {
                    "seq1": seq1,
                    "seq2": seq2,
                    "avg_length": (len(seq1) + len(seq2)) / 2,
                }
                for seq1, seq2 in sorted_pairs
            ]
        }

        with output_file.open("w") as f:
            json.dump(sequence_data, f, indent=4)

        logging.info(
            f"Generated and saved {len(sequence_data['combinations'])} combinations to {output_file}"
        )

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


sequences = load_data()
generate_combinations(sequences)

from Bio import SeqIO
from pathlib import Path
from Algorithms.smith_waterman import smith_waterman

# Load sequences from the FASTA file
fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

# Ensure there are at least 100 sequences
if len(sequences) < 100:
    raise ValueError("The FASTA file must contain at least 100 sequences.")

# Open a file to save the alignment results
output_file = 'Result/smith_waterman_alignment_results.txt'

with open(output_file, 'w') as f:
    # Perform pairwise alignment for the first 100 sequences
    for i in range(100):
        for j in range(i + 1, 100):
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

print(f"Alignment results have been saved to '{output_file}'.")

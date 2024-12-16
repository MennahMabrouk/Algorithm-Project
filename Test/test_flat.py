from Bio import SeqIO
from Algorithms.flat import flat_algorithm

fasta_path = 'Dataset/sequence.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

if len(sequences) < 2:
    raise ValueError("The FASTA file must contain at least two sequences.")

fragment_length = 50
max_iterations = 20

sequence1, sequence2 = sequences[0], sequences[1]
score, fragments = flat_algorithm(sequence1, sequence2, fragment_length, max_iterations)

output_path = 'result/flat_results.txt'
with open(output_path, 'w') as f:
    f.write(f"FLAT Alignment Results:\n")
    f.write(f"Best Score: {score}\n")
    if fragments:
        f.write(f"Fragments aligned: Sequence1[{fragments[0]}:{fragments[0] + fragment_length}] and "
                f"Sequence2[{fragments[1]}:{fragments[1] + fragment_length}]\n")

print(f"Results saved in {output_path}")

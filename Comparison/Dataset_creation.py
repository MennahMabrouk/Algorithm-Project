import random
import json

def generate_sequence(length):
    """Generate a random protein-like sequence of given length."""
    return ''.join(random.choices("WKPVSNTAYYYTFNEMKSADSIYVFFSSNFFKQMVNLGISDINTKDLFNFRFQNTTSPESGWYEFSTSNT", k=length))

def introduce_similarity(sequence, similarity):
    """Introduce a similarity percentage to a given sequence."""
    sequence_length = len(sequence)
    similar_length = int(sequence_length * similarity)
    different_length = sequence_length - similar_length

    similar_part = sequence[:similar_length]
    different_part = ''.join(random.choices("WKPVSNTAYYYTFNEMKSADSIYVFFSSNFFKQMVNLGISDINTKDLFNFRFQNTTSPESGWYEFSTSNT", k=different_length))

    # Shuffle the similar and different parts to create the second sequence
    combined_sequence = list(similar_part + different_part)
    random.shuffle(combined_sequence)
    return ''.join(combined_sequence)

def generate_combinations():
    """Generate 1000 combinations of protein-like sequences with 60-80% similarity."""
    combinations = []

    for _ in range(1000):
        # Randomly choose a length for the first sequence, between 500 and 1000
        length1 = random.randint(500, 1000)
        sequence1 = generate_sequence(length1)

        # Generate a second sequence with a similarity of 60-80% to the first
        similarity = random.uniform(0.6, 0.8)
        sequence2 = introduce_similarity(sequence1, similarity)

        # Add the pair as a tuple to the combinations list
        combinations.append((sequence1, sequence2))

    # Sort combinations by the sum of lengths of the two sequences
    combinations.sort(key=lambda x: (len(x[0]) + len(x[1])))
    return combinations

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    combinations = generate_combinations()
    save_to_json(combinations, "protein_combinations.json")
    print("1000 protein-like sequence combinations with 60-80% similarity have been saved in 'protein_combinations.json'.")

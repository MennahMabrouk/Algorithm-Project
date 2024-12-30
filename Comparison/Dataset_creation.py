import random
import json
from os import path, makedirs

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

    combined_sequence = list(similar_part + different_part)
    random.shuffle(combined_sequence)
    return ''.join(combined_sequence)

def generate_combinations():
    """Generate 1000 combinations of protein-like sequences with 60-80% similarity."""
    combinations = []

    for _ in range(1000):
        length1 = random.randint(500, 1000)
        sequence1 = generate_sequence(length1)
        similarity = random.uniform(0.6, 0.8)
        sequence2 = introduce_similarity(sequence1, similarity)
        combinations.append((sequence1, sequence2))

    combinations.sort(key=lambda x: (len(x[0]) + len(x[1])))
    return combinations

def save_to_json(data, folder, filename):
    """Save data to a JSON file in a specified folder."""
    if not path.exists(folder):
        makedirs(folder)
    file_path = path.join(folder, filename)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data has been saved to {file_path}")

if __name__ == "__main__":
    dataset_folder = path.join("Dataset")
    dataset_file = "protein_combinations.json"
    combinations = generate_combinations()
    save_to_json(combinations, dataset_folder, dataset_file)

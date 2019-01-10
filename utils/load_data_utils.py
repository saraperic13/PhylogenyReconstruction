import numpy as np


def read_data(file_name):
    data = {}
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            if "A" not in line and "C" not in line and "G" not in line and "T" not in line:
                continue
            species, sequence = sequence_to_one_hot_enc(line)
            if len(sequence) <= 0:
                continue

            if species not in data:
                data[species] = []

            data[species].append(np.array(sequence))

    return data, len(data[species])


def sequence_to_one_hot_enc(seq):
    letter_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_seq = []
    species = []
    for i in seq:
        if i in letter_dict:
            one_hot_seq.extend(letter_dict[i])
        else:
            species.append(i)
    species = "".join(species)
    species = species.strip()
    return species, one_hot_seq

import random

import numpy as np


class TrainingDataModel:

    def __init__(self, tree, dna_sequences, dna_sequence_length, dna_num_letters=4,
                 dataset_index=None):
        self.dna_subroots = []
        self.dna_sequences_right_child = []
        self.dna_sequences_left_child = []
        self.are_nodes_together = []
        self.tree = tree
        self.dna_sequence_length = dna_sequence_length
        self.dna_num_letters = dna_num_letters
        self.dna_sequences = dna_sequences
        self.dataset_index = dataset_index
        self.dataset_size = tree.get_number_of_leaves()
        self.descendants_dna_sequences = []

        self.check_and_modify_dataset_index()

    def check_and_modify_dataset_index(self):
        if not self.dataset_index:
            number_of_datasets = len(next(iter(self.dna_sequences.values())))
            self.dataset_index = random.randint(0, number_of_datasets - 1)

    def select_random_subroot(self):
        subroot = self.tree.get_random_node()

        if subroot.name in self.dna_sequences:
            self.dna_subroots.append(self.dna_sequences[subroot.name][self.dataset_index])

        return subroot

    def process_dataset(self, descendants):
        dnas = []
        for child in descendants:
            # TODO extend
            dnas.extend(self.get_dna_of_selected_node(child))

        for i in range(self.dataset_size - len(descendants)):
            # TODO extend
            dnas.extend(
                np.zeros(self.dna_num_letters * self.dna_sequence_length, dtype=np.int64))

        self.descendants_dna_sequences.append(dnas)

    def get_dna_of_selected_node(self, node):
        return self.dna_sequences[node.name][self.dataset_index]

    def process_leaves(self, nodes, together):
        self.are_nodes_together.append(together)
        self.dna_sequences_left_child.append(self.dna_sequences[nodes[0].name][self.dataset_index])
        self.dna_sequences_right_child.append(self.dna_sequences[nodes[1].name][self.dataset_index])
    #
    # def get_dna_subroots(self):
    #     return self.dna_subroots
    #
    # def get_dna_sequences_right_child(self):
    #     return self.dna_sequences_right_child
    #
    # def get_dna_sequences_left_child(self):
    #     return self.dna_sequences_left_child
    #
    # def get_dna_sequences(self):
    #     return self.dna_sequences
    #
    # def get_dataset_index(self):
    #     return self.dataset_index
    #
    # def get_dataset_size(self):
    #     return self.dataset_size
    #
    # def get_are_nodes_together(self):
    #     return self.are_nodes_together
    #
    # def get_dna_sequence_length(self):
    #     return self.dna_sequence_length
    #
    # def get_dna_num_letters(self):
    #     return self.dna_num_letters
    #
    # def get_tree(self):
    #     return self.tree

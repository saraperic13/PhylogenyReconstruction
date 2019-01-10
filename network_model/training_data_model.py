from tree_files import tree_utils

import random

import numpy as np


class TrainingDataModel:

    def __init__(self, tree, dna_sequences, dna_sequence_length, dna_num_letters=4,
                 dataset_index=None):
        self.dna_subroots = []
        self.dna_sequences_node_2 = []
        self.dna_sequences_node_1 = []
        self.node_1 = []
        self.node_2 = []
        self.are_nodes_together = []
        self.tree = tree
        self.dna_sequence_length = dna_sequence_length
        self.dna_num_letters = dna_num_letters
        self.dna_sequences = dna_sequences
        self.tree_index = dataset_index
        # self.dataset_size = tree.get_number_of_leaves()
        self.dataset_size = len(dna_sequences.keys())
        self.descendants_dna_sequences = []

        self.check_and_modify_dataset_index()

    def check_and_modify_dataset_index(self):
        if self.tree_index is None:
            number_of_datasets = len(next(iter(self.dna_sequences.values())))
            self.tree_index = random.randint(0, number_of_datasets - 1)

    def prepare_randomized_batch_sized_data(self, batch_size):
        for i in range(batch_size):
            self.select_random_subroot_and_descendants()

    def prepare_node_pairs(self):
        nodes_names = self.dna_sequences.keys()
        descendants = tree_utils.create_nodes(nodes_name_list=nodes_names)

        descendants_pairs = tree_utils.get_all_node_pairs(nodes_names)
        print(descendants_pairs)
        for pair in descendants_pairs:
            self.process_leaves_v2(pair, [0, 0])
            self.get_descendants_dna_with_padding(descendants)

    def select_random_subroot_and_descendants(self):
        descendants = []
        subroot = self.select_random_subroot()

        tree_utils.get_all_node_descendant_leaves(subroot, descendants)

        self.get_descendants_dna_with_padding(descendants)

        leaves = tree_utils.get_random_descendants(descendants)
        together = tree_utils.are_together(leaves[0], leaves[1], subroot)
        self.process_leaves(leaves, together)

    def select_random_subroot(self):
        subroot = self.tree.get_random_node()

        if subroot.name in self.dna_sequences:
            self.dna_subroots.append(self.dna_sequences[subroot.name][self.tree_index])

        return subroot

    def get_descendants_dna_with_padding(self, descendants):
        dnas = []
        for child in descendants:
            dnas.append(self.get_dna_of_selected_node(child))

        for i in range(self.dataset_size - len(descendants)):
            dnas.append(
                np.zeros(self.dna_num_letters * self.dna_sequence_length, dtype=np.int64))

        self.descendants_dna_sequences.append(dnas)

    def get_dna_of_selected_node(self, node):
        return self.dna_sequences[node.name][self.tree_index]

    def process_leaves(self, nodes, together):
        self.are_nodes_together.append(together)
        self.fill_node_pair_dna_sequences_list([nodes[0].name, nodes[1].name])

    def process_leaves_v2(self, node_names, together):
        self.are_nodes_together.append(together)
        self.fill_node_pair_dna_sequences_list([node_names[0], node_names[1]])

    def fill_node_pair_dna_sequences_list(self, nodes_names_pair):
        node_name_1 = nodes_names_pair[0]
        node_name_2 = nodes_names_pair[1]
        self.node_1.append(node_name_1)
        self.node_2.append(node_name_2)
        self.dna_sequences_node_1.append(self.dna_sequences[node_name_1][self.tree_index])
        self.dna_sequences_node_2.append(self.dna_sequences[node_name_2][self.tree_index])


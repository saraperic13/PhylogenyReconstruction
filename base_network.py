class BaseNetwork:

    def __init__(self, number_of_neurons_per_layer, batch_size, sequence_length, dna_num_letters=4):
        self.number_of_neurons_per_layer = number_of_neurons_per_layer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dna_num_letters = dna_num_letters

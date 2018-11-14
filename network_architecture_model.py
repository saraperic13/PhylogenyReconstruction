class NetworkArchitectureModel:

    def __init__(self, number_of_neurons_encoder, number_of_neurons_per_layer_classifier,
                 batch_size, sequence_length, dna_num_letters=4):
        self.number_of_neurons_encoder = number_of_neurons_encoder
        self.number_of_neurons_per_layer_classifier = number_of_neurons_per_layer_classifier

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dna_num_letters = dna_num_letters

        self.encoder_w = None

    def initialize_encoder(self):
        pass

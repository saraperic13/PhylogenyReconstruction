from network_model.main_network_model import MainNetworkModel

tree_file = "dataset/5000_20.2.tree"
dna_sequence_file = "dataset/seq_5000_20.2.txt"
model_path = "models/5000_30/"

encoder_output_size = 10

feed_forward_hidden_units_1 = 500
feed_forward_hidden_units_2 = 500

sequence_length = 100
number_of_leaves_sequences = 20

dna_num_letters = 4

learning_rate = 0.035

batch_size = 100

num_training_iters = 35

if __name__ == "__main__":
    network = MainNetworkModel(tree_file=tree_file,
                               dna_sequence_file=dna_sequence_file, model_path=model_path,
                               number_of_neurons_per_layer_encoder=[dna_num_letters * sequence_length,
                                                                    encoder_output_size],
                               number_of_neurons_per_layer_classifier=[3 * encoder_output_size,
                                                                       feed_forward_hidden_units_1,
                                                                       feed_forward_hidden_units_2, 2],
                               sequence_length=sequence_length, number_of_leaves_sequences=number_of_leaves_sequences,
                               dna_num_letters=dna_num_letters, batch_size=batch_size,
                               learning_rate=learning_rate,
                               num_training_iters=num_training_iters)
    network.train()

from network_model.test_model import TestModel

tree_file = "dataset/100_5.2.tree"
dna_sequence_file = "dataset/seq_100_5.2.txt"
model_path = "models/fs/"

sequence_length = 100
batch_size = 5

dna_num_letters = 4
number_of_iterations = 3

if __name__ == "__main__":
    test_model = TestModel(tree_file=tree_file, dna_sequence_file=dna_sequence_file, model_path=model_path,
                           sequence_length=sequence_length, batch_size=batch_size, dna_num_letters=dna_num_letters,
                           num_iters_per_tree=number_of_iterations)

    test_model.test()

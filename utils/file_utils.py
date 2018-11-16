def write_to_file(losses, accuracy, avg_loss, avg_acc, path):
    with open(path + "test_loss.txt", "a") as f:
        f.write("\n\n" + avg_loss)
        f.write(losses)

    with open(path + "test_acc.txt", "a") as f:
        f.write("\n\n''" + avg_acc)
        f.write(accuracy)

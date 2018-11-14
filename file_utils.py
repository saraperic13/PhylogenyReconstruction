
def write_to_file(losses, accuracy, avg_loss, avg_acc):
    with open("test/test_loss.txt", "a") as f:
        f.write("\n\n" + avg_loss)
        f.write(losses)

    with open("test/test_acc.txt", "a") as f:
        f.write("\n\n''" + avg_acc)
        f.write(accuracy)

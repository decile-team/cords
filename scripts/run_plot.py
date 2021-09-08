# Please run this script at the root dir of cords
import sys

sys.path.append("../")
sys.path.append("./")

import os
import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", type=str)
args = parser.parse_args()


def plot(x_list, y_list, labels=None, title=None, xlabel=None, ylabel=None, note=None, legend=True):
    if note:
        fig, (ax, note_ax) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.5, 1]})
    else:
        fig, ax = plt.subplots()

    if not (type(x_list[0]) == int or type(x_list[0]) == float):
        for i in range(len(x_list)):
            if labels:
                ax.plot(x_list[i], y_list[i], label=labels[i])
            else:
                ax.plot(x_list[i], y_list[i])
    else:
        ax.plot(x_list, y_list)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.title(title)
    if note:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        note_ax.set_axis_off()
        # note_ax.text(0.5, 0.5, note)
        note_ax.text(0, 0, note)
    if legend:
        ax.legend()

    plt.show()


if __name__ == "__main__":
    # Save results
    save_path = os.path.join(".", "scripts", "RESULTS", args.dir_name)
    with open(os.path.join(save_path, 'save_dict.pickle'), 'rb') as handle:
        save_dict = pickle.load(handle)

    training_args = save_dict["args"]
    n_epochs = save_dict["args"].n_epochs
    train_loss = save_dict["train_loss"]
    train_accu = save_dict["train_accu"]
    valid_loss = save_dict["valid_loss"]
    valid_accu = save_dict["valid_accu"]
    train_elapsed = save_dict["train_elapsed"]

    # Plotting
    plot_title = "%s, %s" % (training_args.dataset, training_args.dss_strategy)

    # Plotting metrics vs epochs
    plot(range(n_epochs), train_loss, title=plot_title, xlabel="epoch", ylabel="training loss")
    plot(range(n_epochs), train_accu, title=plot_title, xlabel="epoch", ylabel="training accuracy")
    plot(range(n_epochs), valid_loss, title=plot_title, xlabel="epoch", ylabel="validation loss")
    plot(range(n_epochs), valid_accu, title=plot_title, xlabel="epoch", ylabel="validation accuracy")

    # Plotting metrics vs training time
    plot(train_elapsed, train_loss, title=plot_title, xlabel="traning time (sec)", ylabel="training loss")
    plot(train_elapsed, train_accu, title=plot_title, xlabel="traning time (sec)", ylabel="training accuracy")
    plot(train_elapsed, valid_loss, title=plot_title, xlabel="traning time (sec)", ylabel="validation loss")
    plot(train_elapsed, valid_accu, title=plot_title, xlabel="traning time (sec)", ylabel="validation accuracy")

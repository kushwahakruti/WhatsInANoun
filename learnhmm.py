import numpy as np
import sys


def read_data(train_input, index_to_word, index_to_flag):
    word_index = {}
    tag_index = {}
    with open(index_to_word) as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        word_index[lines[i].rstrip()] = i
    with open(index_to_tag) as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        tag_index[lines[i].rstrip()] = i
    print(tag_index)
    prior = np.zeros(len(tag_index))
    trans = np.zeros((len(tag_index), len(tag_index)))
    emit = np.zeros((len(tag_index), len(word_index)))
    prior += 1
    trans += 1
    emit += 1

    with open(train_input) as f:
        lines = [line.rstrip().split(" ") for line in f.readlines()]
    for i in range(0, len(lines)):
        if i < 10000:
            line = lines[i]
            for i in range(0, len(line)):
                if i == 0:
                    prior[tag_index[line[i].split("_")[1]]] += 1
                if i != len(line) - 1:
                    trans[tag_index[line[i].split("_")[1]]][tag_index[line[i + 1].split("_")[1]]] += 1
                emit[tag_index[line[i].split("_")[1]]][word_index[line[i].split("_")[0]]] += 1
    prior /= prior.sum()
    trans /= trans.sum(axis=1).reshape(len(tag_index), -1)
    emit /= emit.sum(axis=1).reshape(len(tag_index), -1)

    return prior, trans, emit


if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    prior, trans, emit = read_data(train_input, index_to_word, index_to_tag)

    np.savetxt(hmmprior, prior)
    np.savetxt(hmmtrans, trans)
    np.savetxt(hmmemit, emit)

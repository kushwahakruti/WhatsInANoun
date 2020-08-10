import numpy as np
import sys
import math


def forwardbackward(words, word_index, tag_index, index_tag, prior, trans, emit):
    logalpha = np.zeros((len(words), len(tag_index)))
    beta = np.zeros((len(words), len(tag_index)))
    for j in range(0, len(tag_index)):
        logalpha[0][j] = np.log(prior[j]) + np.log(emit[j][word_index[words[0]]])

    for t in range(1, len(words)):
        for j in range(0, len(tag_index)):
            max_b = -9000000000000000000000000000000000000000000000000000.0
            for k in range(0, len(tag_index)):
                b = np.log(emit[j][word_index[words[t]]]) + logalpha[t - 1][k] + np.log(trans[k][j])
                if max_b < b:
                    max_b = b
                # print(max_b)
            for k in range(0, len(tag_index)):
                logalpha[t][j] += np.exp(np.log(emit[j][word_index[words[t]]]) + logalpha[t - 1][k] + np.log(trans[k][j]) - max_b)
            logalpha[t][j] = np.log(logalpha[t][j]) + max_b

    # print(logalpha)
    max_b = np.max(logalpha[len(logalpha) - 1])
    value = 0
    for i in range(0, len(logalpha[0])):
        value += np.exp(logalpha[len(logalpha) - 1][i] - max_b)
    value = np.log(value) + max_b
    log_likelihood = value

    for j in range(0, len(tag_index)):
        beta[-1][j] = 1

    for t in range(len(words) - 2, -1, -1):
        for j in range(0, len(tag_index)):
            for k in range(0, len(tag_index)):
                beta[t][j] += emit[k][word_index[words[t + 1]]] * beta[t + 1][k] * trans[j][k]

    # prob = alpha * beta
    # prediction_indices = np.argmax(prob, axis=1)
    # prediction = []
    # for i in prediction_indices:
    #     prediction.append(index_tag[i])
    prediction = 0
    return prediction, log_likelihood


def read_data(test_input, index_to_word, index_to_tag, prior, trans, emit, output_file):
    word_index = {}
    tag_index = {}
    index_tag = {}
    predict = []
    average_log_likelihood = 0.0
    total_tags = 0
    correct = 0
    with open(output_file, "w") as f:
        f.write("")
    with open(index_to_word) as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        word_index[lines[i].rstrip()] = i
    with open(index_to_tag) as f:
        lines = f.readlines()
    for i in range(0, len(lines)):
        tag_index[lines[i].rstrip()] = i
        index_tag[i] = lines[i].rstrip()
    with open(test_input) as f:
        lines = [line.rstrip().split(" ") for line in f.readlines()]
    for line in lines:
        words = []
        correct_tags = []
        for word in line:
            words.append(word.split("_")[0])
            correct_tags.append(word.split("_")[1])
        predicted_tags, log_likelihood = forwardbackward(words, word_index, tag_index, index_tag, prior, trans, emit)
        # predicted = ""
        # for word, predicted_tag in zip(line, predicted_tags):
        #     predicted += word.split("_")[0] + "_" + predicted_tag + " "
        # total_tags += len(correct_tags)
        average_log_likelihood += log_likelihood
        # for i in range(0, len(correct_tags)):
        #     if correct_tags[i] == predicted_tags[i]:
        #         correct += 1
        # with open(output_file, "a") as f:
        #     f.write(predicted.strip() + "\n")
    return average_log_likelihood / (float(len(lines)))  # , correct / (float(total_tags))


if __name__ == "__main__":
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    output_file = sys.argv[7]
    metric_file = sys.argv[8]

    prior = np.genfromtxt(hmmprior, delimiter=" ").T
    trans = np.genfromtxt(hmmtrans, delimiter=" ")
    emit = np.genfromtxt(hmmemit, delimiter=" ")

    average_log_likelihood = read_data(test_input, index_to_word, index_to_tag, prior, trans, emit, output_file)
    print(average_log_likelihood)
    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: " + str(average_log_likelihood) + "\n")
        f.write("Accuracy: " + str(accuracy) + "\n")

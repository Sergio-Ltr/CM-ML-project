from activationfunctions import tanh, tanh_prime
from losses import MEE, MEE_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from kfold import KFold
from preprocessing import continuous_standardizer
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import plot_and_save, tr_vl_split, compare
import numpy as np
import matplotlib.pyplot as plot
import time
import pickle


def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    X, Y = load_cup(verbose=output, test=False)
    # train
    # X, n_min, n_max = min_max_normalizer(X)
    # X, means, std = continuous_standardizer(X)
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio = 0.25)
    suffix = "CUP_" + ts
    net = Network("CUP test", MEE, MEE_prime, regularizator = L2)
    net.add(FullyConnectedLayer(10, 20, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(20, 20, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(20, 20, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(20, 20, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(20, 2, initialization_func="normalized_xavier"))
    if (output): net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.01, verbose=output, early_stopping=25, batch_size=1, weight_decay="linear")

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
        if (compare(yvl[i][0][0], out[i][0][0]) and compare(yvl[i][0][1], out[i][0][1])): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100

    if (output): print("Accuracy: {:.4f}%".format(accuracy))

    plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")

test_CUP(output=True)

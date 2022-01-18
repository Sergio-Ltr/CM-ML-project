import numpy as np
import pickle
import random
from utils import training_progress

class Network:
    def __init__(self, name="-unnamed-", loss=None, loss_prime=None, regularizator=None, regularization_l=0, momentum=0, dropout_input=0.3, dropout_hidden=0.5):
        self.name = name  # for logging and output purposes
        self.layers = []  # all the layers will be stored here
        self.loss = loss
        self.loss_prime = loss_prime
        self.regularizator = regularizator
        self.regularization_l = regularization_l
        self.momentum = momentum
        self.dropout_in = dropout_input  # not used
        self.dropout_h = dropout_hidden  # not used


    def summary(self):
        # a summary of the network, for logging and output purposes
        trainable_parameters = 0  # output purposes
        print("Neural Network \"" + self.name + "\"")
        print("+==== Structure")
        for index, layer in enumerate(self.layers):
            print("|\t" + str(index+1) + ". " + type(layer).__name__, end = "")
            try:
                print(" of " +  str(len(layer.weights)) + " -> " + str(layer.weights[0].size) + " units", end = "")
                trainable_parameters += layer.weights.size
                trainable_parameters += layer.bias.size
            except AttributeError:
                pass
            try:
                print(" with " + layer.activation.__name__ + " activation function", end = "")
            except AttributeError:
                pass
            print("")
        if not(self.loss is None):
            print("+==== Loss: " + self.loss.__name__, end="")
        else:
            print("+====")
        if not(self.regularizator is None):
            print(" and " + self.regularizator.__name__ + " regularizator", end="")
        if (self.momentum > 0):
            print(" and momentum = " + str(self.momentum), end="")
        print("")
        print("For a total of " + str(trainable_parameters) + " trainable parameters")


    def set_loss(self, loss, loss_prime):
        # TODO delete this?
        # changes the losses of the net
        self.loss = loss
        self.loss_prime = loss_prime


    def add(self, layer):
        # TODO check input-output dimensions?
        # adds another layer at the bottom of the network
        self.layers.append(layer)


    def predict(self, data):
        # applies the network to the data and returns the computed values
        N = len(data)
        results = []

        for i in range(N):
            output = data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)

        return results


    def training_loop(self, X, Y, X_validation=None, Y_validation=None, epochs=100, learning_rate=0.01, early_stopping=None, batch_size=1, verbose=True):
        N = len(X)
        history = []  # for logging purposes
        M = 0
        if not(X_validation is None): #If validation set is provided
            val_history = []
            M = len(X_validation)
        else:
            val_history = None  # used to check if validation set is present

        es_epochs = 0  # counting early stopping epochs if needed
        min_error = float('inf')
        for i in range(epochs):
            error = 0
            outputs = []
            targets = []
            #shuffle order of inputs each epoch
            temp = list(zip(X, Y))
            random.shuffle(temp)
            X, Y = zip(*temp)
            for j in range(N):
                # compute the output iteratively through each layer
                output = X[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                error += self.loss(Y[j], output)
                # compute the gradient through each layer, while applying
                # the backward propagation algorithm
                outputs.append(output)
                targets.append(Y[j])
                if ((j+1) % batch_size == 0) or (j == N-1):
                    gradient = 0
                    for k in range(len(outputs)):
                        gradient += self.loss_prime(targets[k], outputs[k])
                    gradient /= len(outputs)
                    for layer in reversed(self.layers):
                        gradient = layer.backward_propagation(gradient, learning_rate, self.momentum, self.regularizator, self.regularization_l)
                    outputs = []
                    targets = []
                    #TODO implement variable learning_rate?
                    #eta = (1-alfa)*eta_init + alfa*eta_base
                    #alfa = step/fixed_step

            error /= N  # mean error over the set
            history.append(error)
            if not(val_history is None):
                # if a validation set is given, we now compute the error over it
                val_error = 0
                for j in range(M):
                    output = X_validation[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    val_error += self.loss(Y_validation[j], output)
                val_error /= M
                val_history.append(val_error)
                if (verbose): training_progress(i+1, epochs, suffix=("loss = %f, val_loss = %f" % (error, val_error)))
            else:
                # if no validation set, we simply output the current status
                if (verbose): training_progress(i+1, epochs, suffix=("loss = %f" % (error)))
            if not(early_stopping is None):
                # with early stopping we need to check the current situation and
                # stop if needed
                if not(val_history is None):
                    # we use the validation error if a validation set is given
                    check_error = val_error
                else:
                    # otherwise we just use what we have, the training error
                    check_error = error
                if check_error >= min_error or np.isnan(check_error):
                    # the error is increasing or is stable, or there was an
                    # overflow situation, hence we are going toward an ES
                    es_epochs += 1
                    if es_epochs == early_stopping:
                        if not(val_history is None):
                            if (verbose): print('\nEarly stopping on epoch %d of %d with loss = %f and val_loss = %f' % (i+1, epochs, error, val_error))
                        else:
                            if (verbose): print('\nEarly stopping on epoch %d of %d with loss = %f' % (i+1, epochs, error))
                        break
                else:
                    # we're good
                    es_epochs = 0
                    min_error = check_error
        if (verbose): print("")

        # return the data that we have gathered
        if not(val_history is None):
            return history, val_history
        else:
            return history


    def savenet(self, filename):
        with open(filename, "wb") as savefile:
            pickle.dump(self.__dict__, savefile)


    def loadnet(self, filename):
        with open(filename, "rb") as savefile:
            newnet = pickle.load(savefile)

        self.__dict__.clear()  # clear current net
        self.__dict__.update(newnet)

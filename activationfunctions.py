import numpy as np


def tanh(x):
    # returns the tanh value of x
    return np.tanh(x);


def tanh_prime(x):
    # derivative of the tanh function
    return 1 - np.power(np.tanh(x), 2);


def sigmoid(x):
    # returns the sigmoid value of x
    return 1 / (1 + np.exp(np.dot(-1, x)))  # 1/(1+e^{-x})


def sigmoid_prime(x):
    # derivative of the sigmoid function
    sigma = sigmoid(x)
    return sigma * (1 - sigma)


def relu(x):
    # returns the positive part of x
    return np.maximum(x, 0)


def relu_prime(x):
    # derivative of the ReLU function
    return np.array(x > 0).astype('int')


def leaky_relu(x, leak=0.3):
    # returns the positive part of x or allows a small gradient if x <= 0
    return ((x > 0) * x) + ((x <= 0) * leak * x)


def leaky_relu_prime(x, leak=0.3):
    # derivative of the leay ReLU function
    return (x > 0) + ((x <= 0) * leak)


def softmax(x):
    # return the softmax value of x
    exp = np.exp(x - x.max())  # subtract max for numerical stability
    return exp / exp.sum(axis = 0)


def softmax_prime(x):
    # derivative of the softmax function
    jacobian_m = np.diag(x)
    s = []

    for i in range(len(jacobian_m)):
        s.append([])
        for j in range(len(jacobian_m)):
            if i == j:
                s[i].append(x[i] * (1-x[i]))
            else:
                s[i].append(- x[i] * x[j])
    return np.array(s)
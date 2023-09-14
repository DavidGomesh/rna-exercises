
from sys import setrecursionlimit
setrecursionlimit(100000)

from random import uniform
from activation import step

# Generates a prediction
def predict(inputs, weights, bias, activation):
    return activation(sum(i * w for i, w in zip(inputs, weights)) + bias)

# Fit n epochs
def fit(inputs, targets, weights, bias, learningRate, activation, epochs):
    if epochs == 0:
        return weights, bias
    
    weights, bias = fitEpoch(inputs, targets, weights, bias, learningRate, activation)
    return fit(inputs, targets, weights, bias, learningRate, activation, epochs-1)

# Fit just one epoch
def fitEpoch(inputs, targets, weights, bias, learningRate, activation):
    if not inputs:
        return weights, bias
    
    prediction = predict(inputs[0], weights, bias, activation)
    error = update(targets[0], prediction, learningRate)
    weights = updatedWeights(inputs[0], weights, error)
    bias = updatedBias(bias, error)
    
    return fitEpoch(inputs[1:], targets[1:], weights, bias, learningRate, activation)

# Calculates the update value
def update(target, prediction, learningRate):
    return learningRate * (target - prediction)

# Updates the weights
def updatedWeights(inputs, weights, update):
    return [w + update * i for i, w in zip(inputs, weights)]

# Updates the bias
def updatedBias(bias, update):
    return bias + update

# Perceptron class
class Perceptron:
    def __init__(self, inputs, targets, weights=[], bias=uniform(-1, 1), learningRate=0.01, activation=step):
        self.inputs = inputs
        self.targets = targets
        self.bias = bias
        self.learningRate = learningRate
        self.activation = activation
        self.weights = weights if weights else [uniform(-1, 1) for _ in range(len(inputs[0]))]

    def fit(self, epochs=1):
        self.weights, self.bias = fit(
            self.inputs,
            self.targets,
            self.weights,
            self.bias,
            self.learningRate,
            self.activation,
            epochs
        )
    
    def predict(self, inputs):
        return predict(
            inputs, 
            self.weights, 
            self.bias, 
            self.activation
        )

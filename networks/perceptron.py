
from random import uniform
from activation import step

# Generates a prediction
def predict(inputs, weights, bias, activation):
    return activation(sum(i * w for i, w in zip(inputs, weights)) + bias)

# Fit n epochs
def fit(inputs, targets, weights, bias, learningRate, activation, epochs):
    for _ in range(epochs):
        for input, target in zip(inputs, targets):
            prediction = predict(input, weights, bias, activation)
            error = update(target, prediction, learningRate)
            weights = updatedWeights(input, weights, error)
            bias = updatedBias(bias, error)
    return weights, bias

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
        self.weights = weights if weights else [uniform(-1, 1) for _ in range(len(inputs[0]))]
        self.learningRate = learningRate
        self.activation = activation
        self.targets = targets
        self.inputs = inputs
        self.bias = bias

    def fit(self, epochs=1):
        self.weights, self.bias = fit(
            self.inputs, self.targets, self.weights, self.bias, self.learningRate, self.activation, epochs
        )
    
    def predict(self, inputs):
        return predict(
            inputs,  self.weights,  self.bias,  self.activation
        )

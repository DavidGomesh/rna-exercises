
from sys import setrecursionlimit
setrecursionlimit(10000)

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



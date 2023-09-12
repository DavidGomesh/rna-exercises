
from random import uniform

# Simple Perceptron Class
class Perceptron:
    def __init__(self, inputs, targets, learningRate=0.01, epochs=5000):
        self.inputs = inputs
        self.targets = targets
        self.weights = [uniform(-1, 1) for _ in range(len(inputs[0]))]
        self.bias = uniform(-1, 1)
        self.learningRate = learningRate
        self.epochs = epochs
    
    def fit(self):
        self.weights, self.bias = fit(
            self.inputs, 
            self.targets, 
            self.weights, 
            self.bias, 
            self.learningRate, 
            self.epochs
        )

    def predict(self, inputs):
        return predict(inputs, self.weights, self.bias)


# Generates a prediction
def predict(inputs, weights, bias):
    return [activation(let(inputs, w, bias)) for w in weights]

# Determines if a neuron will be activated or not
def activation(value):
    return 1 if value >= 0 else 0

# Calculates the sum of the inputs multiplied by the weights plus the bias
def let(inputs, weights, bias):
    return sum(i * w for i, w in zip(inputs, weights)) + bias


# Fit n epochs
def fit(inputs, targets, weights, bias, learningRate, epochs):
    if epochs == 0:
        return weights, bias
    
    weights, bias = fitEpoch(inputs, targets, weights, bias, learningRate)
    return fit(inputs, targets, weights, bias, learningRate, epochs - 1)
            
# Fit just one epoch
def fitEpoch(inputs, targets, weights, bias, learningRate):
    if not inputs:
        return weights, bias
    
    prediction = predict(inputs[0], weights, bias)
    if prediction != targets[0]:
        bias = updatedBias(bias, learningRate, targets[0], prediction)
        weights = updatedWeights(inputs[0], weights, bias)

    return fitEpoch(inputs[1:], targets[1:], weights, bias, learningRate)

# Updates the bias
def updatedBias(bias, learningRate, target, prediction):
    return bias + learningRate * (target - prediction)

# Updates the weights
def updatedWeights(inputs, weights, bias):
    return [w + bias * i for i, w in zip(inputs, weights)]


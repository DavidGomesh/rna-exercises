
from random import uniform, randint
from perceptron import fit, predict
from activation import step, identity

inputs = [[randint(0, 9), randint(0, 9)] for _ in range(10)]
targets = [x1 + x2 for x1, x2 in inputs]

EPOCHS = 5000
learningRate = 0.01
weights = [uniform(-1, 1), uniform(-1, 1)]
bias = uniform(-1, 1)
activation = identity

weights, bias = fit(inputs, targets, weights, bias, learningRate, activation, EPOCHS)

print("Final prediction: ", predict([1, 2], weights, bias, activation))
print("Final prediction: ", predict([8, 2], weights, bias, activation))
print("Final prediction: ", predict([5, 5], weights, bias, activation))
print("Final prediction: ", predict([4, 6], weights, bias, activation))

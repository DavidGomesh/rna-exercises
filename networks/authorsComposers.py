
from random import uniform
from perceptron import fit, predict
from perceptron import Perceptron

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

targets = [0, 0, 1, 1]
weights = [0.1, -0.1]
# weights = [uniform(-1, 1) for _ in range(len(inputs[0]))]

w, b = fit(inputs, targets, weights, uniform(-1, 1), 0.01, 1000)

print(predict([0, 0], w, b))
print(predict([0, 1], w, b))
print(predict([1, 0], w, b))
print(predict([1, 1], w, b))

# p = Perceptron(inputs, targets)

# print("Initial Weights: " + str(p.weights))
# p.fit(9000)
# print("Final Weights: " + str(p.weights))

# print("Prediction: " + str(p.predict([0, 0])))
# print("Prediction: " + str(p.predict([0, 1])))
# print("Prediction: " + str(p.predict([1, 0])))
# print("Prediction: " + str(p.predict([1, 1])))


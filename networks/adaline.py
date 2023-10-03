
from numpy import array, dot, mean, square
from numpy.random import uniform
from activation import bipolar, identity, sign

class Adaline:
    def __init__(self, inputs, targets, learningRate=0.01, precision=0.0001, activation=bipolar):
        self.inputs = array(inputs)
        self.targets = array(targets)
        self.weights = uniform(-1, 2, len(inputs[0]))
        self.bias = uniform(-1, 2)
        self.learningRate = learningRate
        self.precision = precision
        self.activation = activation

    # Generate a prediction
    def predict(self, inputs):
        return self.activation(dot(inputs, self.weights) + self.bias)

    # Calculate the Mean Squared Error
    def meanSquaredError(self):
        return mean(square(self.targets - [self.predict(i) for i in self.inputs]))

    # Train the network
    def fit(self, epochs):
        for _ in range(epochs):
            oldMse = self.meanSquaredError()

            # if oldMse == 0:
            #     return

            print("BEFORE TRAINING: ", _ + 1, "\t Weights: ", self.weights, "\tBias: ", self.bias, "\tMSE: ", self.meanSquaredError())
            for i, t in zip(self.inputs, self.targets):
                # print("\nWeights: ", self.weights, "\tBias: ", self.bias)
                prediction = self.predict(i)
                error = t - prediction
                self.weights += self.learningRate * error * i
                self.bias += self.learningRate * error
                # print("Weights: ", self.weights, "\tBias: ", self.bias)
                # print("Inputs: ", i, "\tTarget: ", t, "\tPrediction:  ", prediction, "\tError: ", error, "\tOld MSE: ", oldMse, "\tCur. MSE: ", self.meanSquaredError())

            print("AFTER  TRAINING: ", _ + 1, "\t Weights: ", self.weights, "\tBias: ", self.bias, "\tMSE: ", self.meanSquaredError())
            print()
            curMse = self.meanSquaredError()
            # if abs(curMse - oldMse) <= self.precision:
            #     return









# Weights:  [ 73.77124377 123.39900389 -36.25782475]     Bias:  151.7494412326395



inputs = [[-0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.085, 0.6876, 12.071], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.673, 8.3265], [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.466, 5.4323], [0.434, 0.687, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.485], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115, -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.797, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.22, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]]
targets = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1]

a = Adaline(inputs, targets, learningRate=0.001, activation=bipolar)
a.fit(300)

for i, t in zip(inputs, targets):
    print("Inputs: ", i, "\tTarget: ", t, "\tPredict: ", a.predict(i))
    assert a.predict(i) == t

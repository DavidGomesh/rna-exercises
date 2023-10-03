
from random import uniform

class Adaline:
    def __init__(self, inputs, targets, learningRate, precision, activation):
        self.inputs = inputs
        # self.inputs = [[1, 1, 1], [1, 1, 1]]
        self.targets = targets
        # self.targets = [-1, -1]
        self.weights = [uniform(-1, 1) for _ in range(len(inputs[0]))]
        # self.weights = [0, 0, 0]
        self.bias = uniform(-1, 1)
        # self.bias = 0
        self.learningRate = learningRate
        self.precision = precision
        self.activation = activation

    def predict(self, inputs):
        return self.activation(sum(i * w for i, w in zip(inputs, self.weights)) - self.bias)
    
    def meanSquaredError(self) -> float:
        return sum(pow(t - self.predict(i), 2) for i, t in zip(self.inputs, self.targets)) / len(self.inputs)
    
    def fit(self, epochs):
        for _ in range(epochs):
            oldMse = self.meanSquaredError()

            for i, t in zip(self.inputs, self.targets):
                prediction = self.predict(i)
                error = t - prediction
                self.weights = [w + self.learningRate * error * i for i, w in zip(i, self.weights)]
                self.bias += self.learningRate * error
                print("Inputs: ", i, "\tTarget: ", t, "\tPrediction:  ", prediction, "\tError: ", error, "\tMSE: ", self.meanSquaredError())

            currMse = self.meanSquaredError()
            print("MSE Cur:", currMse, "Old: ", oldMse, "Stop: ", abs(currMse - oldMse) <= self.precision, "\n")
            # if abs(currMse - oldMse) <= self.precision:
            #     return



    

EPOCHS = 10
precision = 0.0001
learningRate = 0.01
activation = lambda x: 1 if x >= 0 else -1
# activation = lambda x: x
inputs2 = [[-0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.085, 0.6876, 12.071], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.673, 8.3265], [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.466, 5.4323], [0.434, 0.687, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.485], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115, -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.797, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.22, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]]
targets = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1]

# inputs2 = [[-0.6508, 0.1097, 4.0009]]
# targets = [-1]

a = Adaline(inputs2, targets, learningRate, precision, activation)

a.fit(EPOCHS)
# print(a.predict([-0.6508, 0.1097, 4.0009]))
# print(a.predict([-0.6508, 0.1097, 4.0009]))
print(a.meanSquaredError())




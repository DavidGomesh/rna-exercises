
from perceptron import Perceptron
from perceptron import predict

from sys import setrecursionlimit

# Recursion limit: 10.000
setrecursionlimit(10**4)

with open('dataset.txt', 'r') as file:
    next(file)
    inputs = []
    targets = []

    for line in file:
        x1, x2, x3, t = line.strip().split()
        inputs.append([float(x1), float(x2), float(x3)])
        targets.append(float(t))

# print(inputs)
# print(targets)

# p = Perceptron(inputs, targets, learningRate=1, epochs=5000)
# p.fit()

print(predict([1, 2, 3], [[1, 2, 3], [9, 8, 7]], 233))

# p.weights = [3.976082752263153, 73.3207516640089, -4.936692759557548] 
# p.bias = 0.07919377774721746

# assert p.predict([-0.6508, 0.1097, 4.0009]) == -1.0000
# assert p.predict([-1.4492, 0.8896, 4.4005]) == -1.0000
# assert p.predict([2.0850, 0.6876, 12.0710]) == -1.0000
# assert p.predict([0.2626, 1.1476, 7.7985]) == 1.0000
# assert p.predict([0.6418, 1.0234, 7.0427]) == 1.0000
# assert p.predict([0.2569, 0.6730, 8.3265]) == -1.0000
# assert p.predict([1.1155, 0.6043, 7.4446]) == 1.0000
# assert p.predict([0.0914, 0.3399, 7.0677]) == -1.0000
# assert p.predict([0.0121, 0.5256, 4.6316]) == 1.0000
# assert p.predict([-0.0429, 0.4660, 5.4323]) == 1.0000
# assert p.predict([0.4340, 0.6870, 8.2287]) == -1.0000
# assert p.predict([0.2735, 1.0287, 7.1934]) == 1.0000
# assert p.predict([0.4839, 0.4851, 7.4850]) == -1.0000
# assert p.predict([0.4089, 0.1267, 5.5019]) == -1.0000
# assert p.predict([1.4391, 0.1614, 8.5843]) == -1.0000
# assert p.predict([-0.9115, 0.1973, 2.1962]) == -1.0000
# assert p.predict([0.3654, 1.0475, 7.4858]) == 1.0000
# assert p.predict([0.2144, 0.7515, 7.1699]) == 1.0000
# assert p.predict([0.2013, 1.0014, 6.5489]) == 1.0000
# assert p.predict([0.6483, 0.2183, 5.8991]) == 1.0000
# assert p.predict([-0.1147, 0.2242, 7.2435]) == -1.0000
# assert p.predict([-0.7970, 0.8795, 3.8762]) == 1.0000
# assert p.predict([-1.0625, 0.6366, 2.4707]) == 1.0000
# assert p.predict([0.5307, 0.1285, 5.6883]) == 1.0000
# assert p.predict([-1.2200, 0.7777, 1.7252]) == 1.0000
# assert p.predict([0.3957, 0.1076, 5.6623]) == -1.0000
# assert p.predict([-0.1013, 0.5989, 7.1812]) == -1.0000
# assert p.predict([2.4482, 0.9455, 11.2095]) == 1.0000
# assert p.predict([2.0149, 0.6192, 10.9263]) == -1.0000
# assert p.predict([0.2012, 0.2611, 5.4631]) == 1.0000

# print(p.weights, p.bias)

inputs = [
    [-0.3665, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548],
]






# from perceptron import Perceptron
# from perceptron import predict
# from perceptron import fit2

# inputs = [
#     [0.1, 0.4, 0.7],
#     [0.5, 0.7, 0.1],
#     [0.6, 0.9, 0.8],
#     [0.3, 0.7, 0.2],
# ]

# targets = [1, 1, -1, -1]

# learningRate = 0.05

# otherInputs = [
#     [0.2, 0.3, 0.6], 
#     [0.5, 0.8, 0.8], 
#     [0.3, 0.6, 0.4], 
#     [0.2, 0.6, 0.3]
# ]

# otherTargets = [1, -1, -1, -1]

# def test01():
#     EPOCHS = 38
#     stepFn = lambda u: 1 if u >= 0 else -1
#     # weights = [-6, 0.8, -8.6, ]
#     # bias = -3.6

#     p = Perceptron(inputs=inputs, targets=targets, learningRate=learningRate, activation=stepFn)
#     p.fit(EPOCHS)

#     testInputs(p)

# def test02():
#     EPOCHS = 54
#     weights = [-0.241, 0.68, -0.778, 0.015]

# def testInputs(p: Perceptron):
#     for i, t in zip(inputs + otherInputs, targets + otherTargets):
#         print(f"Inputs: {i}\tTarget: {t}\tPrediction: {p.predict(i)}")


# # test01()

# # fit2(inputs, targets, )


# # for i, t in zip(inputs + otherInputs, targets + otherTargets):
# #     print(f"I: {i}\tT: {t}\tP: {predict(i, [0.8, -8.6, -3.6], -6, lambda u: 1 if u >= 0 else -1)}")


print(len([1, 2]))







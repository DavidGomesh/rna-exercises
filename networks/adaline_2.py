import numpy as np

class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def fit(self, inputs, targets):
        for epoch in range(self.epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = targets[i] - prediction
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

                # opcional: imprimir o erro a cada época
                if i == len(inputs) - 1:
                    mse = np.mean(np.square(targets - self.predict(inputs)))
                    print(f"Epoch {epoch+1}/{self.epochs}, Mean Squared Error: {mse}")

# Exemplo de uso
# inputs = np.array([[0.1, 0.4, 0.7], [0.3, 0.7, 0.2], [0.6, 0.9, 0.8], [0.5, 0.7, 0.1]])
# targets = np.array([1, -1, -1, 1])

# # Inicializa e treina o Adaline
adaline = Adaline(input_size=3, learning_rate=0.01, epochs=100)
adaline.train(inputs, targets)

# # Realiza uma previsão após o treinamento
# print("Previsão para [0.1, 0.4, 0.7]:", adaline.predict([0.1, 0.4, 0.7]))


# print(np.array([1, 2, 3]).size)


import numpy as np

class Adaline(object):
    def __init__(self, taxaAprendizado=0.01, precisao=0.00001, epocas=50, bias=-1):
        self.taxaAprendizado = taxaAprendizado
        self.precisao = precisao
        self.epocas = epocas
        self._bias = bias
        self.numEpocas = 0
        print(self._bias)
        print(self.epocas)

    def _ativacao(self, valor):
        if valor > 0.30 :
            return 1
        else:
            return -1

    def fit(self, X, y):
        self._pesos = np.random.uniform(0, 1, (X.shape[1]))
        print(self._pesos)
        self._MSE = []
        numEpocas = 0
        parar = False

        while not parar:
            EQMAnterior = self.EQM(X, y)

            for i in range(len(X)):
                soma = self._bias*(-1)
                for j in range(len(X[i])):
                    soma += self._pesos[j]*X[i][j]

                self._bias += self.taxaAprendizado * (y[i] - soma) * (-1)
                for j in range(len(X[i])):
                    self._pesos[j] += self.taxaAprendizado * (y[i] - soma) * X[i][j]

            numEpocas += 1
            self.numEpocas = numEpocas
            self._MSE.append(EQMAnterior)
            if abs(self.EQM(X, y) - EQMAnterior) <= self.precisao:
                parar = True
            if(numEpocas >= self.epocas):
                parar = True


    def EQM(self, X, y):
        EQM = 0
        for i in range(len(X)):
            soma = self._bias*(-1)
            for j in range(len(X[i])):
                soma += self._pesos[j] * X[i][j]
        EQM += (y[i] - soma)**2

        return EQM/len(X)

    def testa(self, X, t):
        for i in range(len(X)):
            soma = self._bias*(-1)
            for j in range(len(X[i])):
                soma += self._pesos[j]*X[i][j]

            saida = self._ativacao(soma)
            print("I: ", X[i], "\tT: ", t[i])
            assert saida == t[i]

            # if saida == -1:
            #     print('-1')
            # if saida == 1:
            #     print('1')



# Dados de entrada e saída desejada
inputs = [[-0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.085, 0.6876, 12.071], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.673, 8.3265], [1.1155, 0.6043, 7.4446], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.466, 5.4323], [0.434, 0.687, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.485], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115, -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.797, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.22, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9263], [0.2012, 0.2611, 5.4631]]
targets = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1]

# Criar e treinar o modelo Adaline
adaline = Adaline()
adaline.fit(np.array(inputs), np.array(targets))

# Testar o modelo com os dados de entrada
print("Resultados do teste:")
adaline.testa(np.array(inputs), targets)
print("Número de épocas:", adaline.numEpocas)

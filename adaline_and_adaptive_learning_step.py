import random
import numpy as np
import matplotlib.pyplot as plt

def paint_Graph(adalineItersMas = [],adalineErrors = [],AdaptiveStepTrainingItersMas= [],AdaptiveStepTrainingErrors = []):
    fig, ax = plt.subplots()
    ax.plot(adalineItersMas, adalineErrors)
    ax.plot(AdaptiveStepTrainingItersMas,AdaptiveStepTrainingErrors)
    ax.grid()
    ax.legend(["Последовательное обучение","Адаптивный шаг обучения"], loc = "upper right")
    ax.set(xlabel='Число итераций', ylabel='Число ошибок', title='Зависимость Ошибок от Числа итераций')
    plt.show()

class adalineAlgorithm(object):

    def __init__(self, numInputElements=3, numOutputElements=3, alpha='', minError='', maxIterations='', outputElements=[], weights=[]):
        self.numInputElements = numInputElements
        self.numOutputElements = numOutputElements
        self.alpha = alpha
        self.minError = minError
        self.maxIterations = maxIterations
        self.Es = 0
        self.errors = list()
        self.outputElements = outputElements
        self.weights = self.initWeights()
        print(self.weights)
        self.itersMas = list()

    def initWeights(self):
            weights = []
            for randomWeights in range(0, self.numOutputElements):
                weights.append(random.random())
            return weights

    def procces(self, trainginData, iters):
        output = 0
        for j in range(0, len(trainginData[iters])):
                output += trainginData[iters][j] * self.weights[j]
        return output

    def train(self, trainginData):
        for iters in range(1, self.maxIterations):
            print("Iterate" + str(iters))
            output = 0
            for i in range(0, len(trainginData)):
                output = self.procces(trainginData, i)
                desiredOutput = output
                print(output)
                for data in range(0, len(self.weights)):
                    self.weights[data] = self.weights[data] + self.alpha * (self.outputElements[i] - desiredOutput) * trainginData[i][data]
                self.Es = self.Es + (self.outputElements[i] - desiredOutput)**2
            self.Es = 0.5 * self.Es
            self.errors.append(np.around(self.Es, decimals=4))
            self.itersMas.append(iters)
            print("Es" + str(self.Es))
            if self.Es < self.minError:
                break

    def print_weights(self):
        print(self.weights)
        
    def getItersMas(self):
        return self.itersMas
    
    def getErrors(self):
        return self.errors

        
class AdaptiveStepTraining(object):

    def __init__(self, numInputElements=3, numOutputElements=3, alpha='', minError='', maxIterations='', outputElements=[], weights=[]):
        self.numInputElements = numInputElements
        self.numOutputElements = numOutputElements
        self.alpha = alpha
        self.minError = minError
        self.maxIterations = maxIterations
        self.Es = 0
        self.errors = list()
        self.outputElements = outputElements
        self.weights = self.initWeights()
        print(self.weights)
        self.itersMas = list()
        self.SumTrainginDataIter = 0

    def initWeights(self):
            weights = []
            for randomWeights in range(0, self.numOutputElements):
                weights.append(random.random())
            return weights

    def procces(self, trainginData, iters):
        output = 0
        for j in range(0, len(trainginData[iters])):
                output += trainginData[iters][j] * self.weights[j]
                self.SumTrainginDataIter = self.SumTrainginDataIter + trainginData[iters][j]
        return output

    def train(self, trainginData):
        for iters in range(1, self.maxIterations):
            print("Iterate" + str(iters))
            output = 0
            for i in range(0, len(trainginData)):
                output = self.procces(trainginData, i)
                e = output
                print(output)
                self.alpha = 1 / (1 + self.SumTrainginDataIter * self.SumTrainginDataIter)
                for data in range(0, len(self.weights)):
                    self.weights[data] = self.weights[data] + self.alpha * (self.outputElements[i] - e) * trainginData[i][data]
                self.Es = self.Es + (self.outputElements[i] - e)**2
                self.SumTrainginDataIter = 0
            self.Es = 0.5 * self.Es
            self.errors.append(np.around(self.Es, decimals=4))
            self.itersMas.append(iters)
            print("Es" + str(self.Es))
            if self.Es < self.minError:
                break

    def print_weights(self):
        print(self.weights)
    
    def getItersMas(self):
        return self.itersMas
    
    def getErrors(self):
        return self.errors
    
trainginData = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
#outputElements = [0, -11, -17, -28, 8, -3, -9, -20]
#outputElements = [0, -3, 5, 2, 4, 1, 9, 6]
outputElements = [0, -54, 1, -53, 32, -22, 33, -21]
#outputElements =[0,-1446241,11111,-1435130,111345,-1334896,122456,-1323785]

alpha = 0.099
minError = 0.0001
maxIterations = 10000
adaline = adalineAlgorithm(outputElements=outputElements, alpha=alpha, minError=minError, maxIterations=maxIterations)
adaline.train(trainginData)
adaline.print_weights()
AdaptiveStepTraining = AdaptiveStepTraining(outputElements=outputElements, alpha=alpha, minError=minError, maxIterations=maxIterations)
AdaptiveStepTraining.train(trainginData)
AdaptiveStepTraining.print_weights()
adalineItersMas = adaline.getItersMas()
adalineErrors = adaline.getErrors()
AdaptiveStepTrainingItersMas = AdaptiveStepTraining.getItersMas()
AdaptiveStepTrainingErrors = AdaptiveStepTraining.getErrors()
paint_Graph(adalineItersMas = adalineItersMas,adalineErrors = adalineErrors,AdaptiveStepTrainingItersMas = AdaptiveStepTrainingItersMas,AdaptiveStepTrainingErrors = AdaptiveStepTrainingErrors)


import numpy as np 
import random

def loadDataSet(fileName):

    fr = open(fileName)
    arrayOfData = fr.readlines()

    labelList = []
    parameterMatrix = []

    for dataString in arrayOfData:
        data = dataString.rstrip().split('\t')
        data = list(map(float, data))

        label = data[-1]
        labelList.append(label)
        del data[-1]
        parameterMatrix.append(data)

    return np.array(parameterMatrix), np.array(labelList)

def selectFromRange(lowerBoundary, upperBoundary):
    randomValue1 = random.randint(lowerBoundary, upperBoundary)
    randomValue2 = random.randint(lowerBoundary, upperBoundary)
    while (randomValue2 == randomValue1):
        randomValue2 = random.randint(lowerBoundary, upperBoundary)
    return randomValue1, randomValue2

def randomlyGenerateAlpha(lowerBoundary, upperBoundary, numberOfAlpha):

    alpha = np.zeros(numberOfAlpha)
    for i in range(numberOfAlpha):
        alpha[i] = random.uniform(lowerBoundary, upperBoundary)
    return alpha


def limitBoundary(inputValue, lowerBoundary, upperBoundary):
    if inputValue > upperBoundary:
        return upperBoundary
    elif inputValue < lowerBoundary:
        return lowerBoundary
    else:
        return inputValue


def SMO(dataParameterMatrix, labelList, C, tolerance, maxIteration):
    N, D = np.shape(dataParameterMatrix)
    alpha = np.zeros()
    for iteration in range(maxIteration):
        i, j = selectFromRange(0, N-1)
        alpha_i = alpha[i]
        LAMBDA  = - (np.dot(alpha, labelList.T) - alpha[i] * labelList[i] - alpha[j] * labelList[j])
        alpha_j = labelList[j] * (LAMBDA - alpha[i] * labelList[i])



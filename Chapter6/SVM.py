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

def selectAlphaIndex(i, IdxLowerBoundary, IdxUpperBoundary):
    j = random.randint(IdxLowerBoundary, IdxUpperBoundary)
    while (i == j):
        j = random.randint(IdxLowerBoundary, IdxUpperBoundary)
    return j



def clipAlpha(alphaUnclipped, lowerBoundary, upperBoundary):
    if alphaUnclipped >= upperBoundary:
        return upperBoundary
    elif alphaUnclipped <= lowerBoundary:
        return lowerBoundary
    else:
        return alphaUnclipped


def SMO(dataParameterMatrix, labelList, alphaUpperBoundary, tolerance, maxIteration):
    x = dataParameterMatrix
    y = labelList
    b = 0
    N, D = np.shape(x)
    alpha = np.zeros(N)

    iteration = 1
    while iteration <= maxIteration:

        for i in range(N):

            mediumMatrix_1 = (alpha * y).reshape(len(alpha * y), 1) * x     # (N, D)
            w = np.sum(mediumMatrix_1, axis = 0)    # (D,) this matrix is actually w in formula 21 in SVM.docx
            predictValue_i = np.dot(w, x[i]) + b    # a scalar
            error_i = predictValue_i - y[i]

            if error_i * y[i] < -tolerance or error_i * y[i] > tolerance:

                j = selecAlphaIndex(i, 0, N-1)

                Kii = np.dot(x[i], x[i].T)
                Kij = np.dot(x[i], x[j].T)
                Kjj = np.dot(x[j], x[j].T)
                s = y[i] * y[j]
                LAMBDA = - (np.sum(alpha * y) - (alpha * y)[i] - (alpha * y)[j])

                mediumMatrix_2 = w - mediumMatrix_1[i] - mediumMatrix_1[j]     # (D, )

                Ki = y[i] * np.dot(x[i], mediumMatrix_2.T)
                Kj = y[j] * np.dot(x[j], mediumMatrix_2.T)

                if y[i] == y[j]:
                    B = y[j] * LAMBDA * Kjj - y[i] * LAMBDA * Kij + Kj - Ki
                    A = s * Kij - 0.5 * Kii - 0.5 * Kjj 
                    L = max(alpha[i] + alpha[j] - alphaUpperBoundary, 0)
                    H = min(alpha[i] + alpha[j], alphaUpperBoundary)
                else:
                    B = - y[j] * LAMBDA * Kjj - y[i] * LAMBDA * Kij + 2 - Kj - Ki
                    A = - s * Kij - 0.5 * Kii - 0.5 * Kjj
                    L = max(alpha[i] - alpha[j], 0)
                    H = min(alphaUpperBoundary + alpha[i] - alpha[j], alphaUpperBoundary)

                if A != 0:
                    alphaUnclipped = - B / (2 * A)
                    alpha[i] = clipAlpha(alphaUnclipped, L, H)
                else:
                    if B > 0:
                        alpha[i] = H
                    else:
                        alpha[i] = L

                if y[i] == y[j]:
                    alpha[j] = y[j] * LAMBDA - alpha[i]
                else:
                    alpha[j] = y[j] * LAMBDA + alpha[i]

                alphaPairsChange += 1

        if alphaPairsChange == 0:
            iteration += 1
        else:
            iteration = 0

    b = 1




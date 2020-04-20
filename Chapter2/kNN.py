import numpy as np
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return group, labels

  
def classify0(inX, dataSet, labels, k):


    # step 1
    dataSetSize = dataSet.shape[0]
    differenceMatrix = np.tile(inX, (dataSetSize, 1)) - dataSet
    L2Distances = np.sum(np.square(differenceMatrix), axis = 1)
    # step 2
    sortedL2DistanceIndices = np.argsort(L2Distances)
    # step 3
    kNeareatNeighbor = sortedL2DistanceIndices[0:k]
    # step 4
    kNNLabels = {}
    for i in kNeareatNeighbor:
        if labels[i] not in kNNLabels.keys():
            kNNLabels[labels[i]] = 1
        else:
            kNNLabels[labels[i]] += 1
    # step 5
    maxValue = 0
    for j, k in kNNLabels.items():
        if k > maxValue:
            maxLabel = j
            maxValue = k

    return maxLabel


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numOfLines = len(arrayOfLines)        # arrayOfLines is a list, every element is a line which includes 4 items
    matrix = np.zeros((numOfLines, 3))    # parameter matrix
    labelVector = []                      # label vector
    i = 0
    for line in arrayOfLines:
        line = line.strip()               # remove the spaces in the beginning and end
        listFromLines = line.split('\t')  # split the list based on tab
        matrix[i,:] = listFromLines[0:3]
        labelVector.append(listFromLines[3])
        i += 1

    return matrix, labelVector

def normalization(dataSet):
    maxValueMatrix = np.max(dataSet, axis = 0, keepdims = True)
    minValueMatrix = np.min(dataSet, axis = 0, keepdims = True)
    ranges = maxValueMatrix - minValueMatrix
    normDataSet = (dataSet - minValueMatrix) / ranges
    return normDataSet, ranges, minValueMatrix


def datingClassTest(filename):
    matrix, labelVector = file2matrix(filename)
    normMatrix, _, _ = normalization(matrix)

    ratio = 0.1
    numTestSet = int(normMatrix.shape[0] * ratio)
    numDataSet = normMatrix.shape[0] - numTestSet

    dataMatrix = normMatrix[0:numDataSet, :]
    testMatrix = normMatrix[numDataSet:(numDataSet + numTestSet), :]
    dataLabels = labelVector[0:numDataSet]
    testLabels = labelVector[numDataSet:(numDataSet + numTestSet)]

    errorCounter = 0
    for i in range(numTestSet):
        classifyResult = classify0(testMatrix[i, :], dataMatrix, dataLabels, 3)
        if classifyResult != testLabels[i]:
            errorCounter += 1
        print('Classified result is %s, correct result is %s' % (classifyResult, testLabels[i]))
    print('Total error rate is: %f' %(errorCounter / numTestSet))


def img2vec(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()

    vector = []
    for line in arrayOfLines:
        line = line.strip()
        for characher in line:
            vector.append(int(characher))
    vector = np.array(vector)

    return vector
    

def handWritingClassifyTest():
    trainingFileList = listdir('trainingDigits')
    numOfFiles = len(trainingFileList)
    dimensions = len(img2vec('trainingDigits/%s' % trainingFileList[0]))

    trainingMatrix = np.zeros((numOfFiles, dimensions))
    trainingLabels = []
    i = 0
    for trainingFile in trainingFileList:
        trainingMatrix[i, :] = img2vec('trainingDigits/%s' % trainingFile)
        trainingFile = trainingFile.split('_')
        trainingLabels.append(trainingFile[0])   # trainingFile[0] is a string
        i += 1

    testFileList = listdir('testDigits')
    k = 3
    correctCount = 0
    totalCount = len(testFileList)
    for testFile in testFileList:
        testVector = img2vec('testDigits/%s' % testFile)
        classifyLabel = classify0(testVector, trainingMatrix, trainingLabels, k)
        testFile = testFile.split('_')
        correctLabel = testFile[0]
        print('Classified result is %s, correct result is %s' % (classifyLabel, correctLabel))
        if classifyLabel == correctLabel:
            correctCount += 1
    errorRate = (totalCount - correctCount) / totalCount
    print('Total error rate is: %f' % errorRate)
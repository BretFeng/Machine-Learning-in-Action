import numpy as np

def loadDataSet(filename):
    """
    Load the data set in the file, and transform those data into two matrices.
    One matrix is the data parameter matrix consists of all data's (all rows)
    all parameters (all columns but the last column) plus the bias terms. 
    Another matrix is the category matrix consists of all data's (all rows) 
    categories (the last column).

    Input:
    - filename: the input text file name

    Return:
    - parameterMatrix: (N, D+1), a matrix consists of all data's (all rows) all parameters
                       (all columns but the last column, the last column is categories). 
                       Then plus the bias terms in the first column.
    - category: (N,), a matrix consists of all data's (all rows) categories (the last column)

    """
    fr = open(filename)
    arrayOfData = fr.readlines()

    N = len(arrayOfData)                            # sample number
    sample0 = arrayOfData[0].rstrip().split('\t')
    D = len(sample0) - 1                            # parameter (dimention) number

    parameterMatrix = np.zeros((N, D))
    category = np.zeros(N)

    for i in range(N):
        testData = arrayOfData[i]
        testData = testData.rstrip()
        paraOfData = testData.split('\t')           # a list of strings
        paraOfData = list(map(float, paraOfData))   # change the items in the list from strings to float data

        parameterMatrix[i] = paraOfData[0:len(paraOfData)-1]
        category[i] = paraOfData[-1]

    biasTerms = np.ones((N, 1))                     # the bias terms
    parameterMatrix = np.column_stack((biasTerms, parameterMatrix))   # includes the bias terms in the first column

    return parameterMatrix, category

def sigmoid(z):
    """
    the sigmoid function

    Input:
    - z: input value

    Return:
    - y: the output value of sigmoid function with input value z

    """
    y = 1 / (1 + np.exp(-z))

    return y

def gradientDescent(parameterMatrix, category, step, iterationNumber):
    """
    a function that uses gradient descent to train network
    

    Input:
    - parameterMatrix: (N, D+1), a matrix consists of all data's (all rows) all parameters (all columns) plus the bias terms
    - category: (N,), a matrix consists of all data's categories
    - step: a scalar which represents the step length (value) while gradient descent
    - iterationNumber: a scalar that shows the time of iteration

    Return:
    - weights: (D+1, 1), the final weights values after training (iteration)
    - weightsHistory: (D+1, iterNumber), all weights values during each iteration

    """   

    import random
    sampleNumber = len(parameterMatrix)          # N
    dimensionNumber = len(parameterMatrix[0])    # D+1

    weights = []
    for i in range(dimensionNumber):
        weights.append(random.uniform(0,1))                 # radomly initialize the weight initial values
    weights = np.array(weights)                             # (D+1,)
    weights = np.reshape(weights, (dimensionNumber, 1))     # (D+1, 1)

    weightsHistory = np.zeros((dimensionNumber, iterationNumber))   # (D+1, iterNumber)

    category = np.array(category)                           # (N,)
    

    for j in range(iterationNumber):
        z = np.dot(parameterMatrix, weights)                         # (N, 1)
        y = sigmoid(z)                                               # (N, 1)
        error = y - category.reshape(len(category), 1)               # (N, 1)
        weights = weights - step * np.dot(parameterMatrix.T, error)  # (D+1, 1)
        weightsHistory[:,j] = weights[:,0]                           # (D+1, 1)

    return weights, weightsHistory


def stocGradientDescent(parameterMatrix, category, step, iterationNumber):
    """
    a function that uses stochastic gradient descent to train network
    

    Input:
    - parameterMatrix: (N, D+1), a matrix consists of all data's (all rows) all parameters (all columns) plus the bias terms
    - category: (N,), a matrix consists of all data's categories
    - step: a scalar which represents the inital step length (value) at gradient descent
    - iterationNumber: a scalar that shows the time of iteration

    Return:
    - weights: (D+1, 1), the final weights values after training (iteration)
    - weightsHistory: (D+1, iterNumber), all weights values during each iteration

    """   

    import random
    sampleNumber = len(parameterMatrix)          # N
    dimensionNumber = len(parameterMatrix[0])    # D+1

    weights = []
    for i in range(dimensionNumber):
        weights.append(random.uniform(0,1))                 # radomly initialize the weight initial values
    weights = np.array(weights)                             # (D+1,)
    weights = np.reshape(weights, (dimensionNumber, 1))     # (D+1, 1)

    weightsHistory = np.zeros((dimensionNumber, iterationNumber))   # (D+1, iterNumber)

    category = np.array(category)                           # (N,)
    
    dynamicStep = step                                      # step will be reduced at every iteration
    for j in range(iterationNumber):
        trainSampleIdx = random.randint(0, sampleNumber - 1)   # [0, N-1]

        z = np.dot(parameterMatrix, weights)                                          # (N, 1)
        y = sigmoid(z)                                                                # (N, 1)
        error = y - category.reshape(len(category), 1)                                # (N, 1)
        weights = weights - dynamicStep * \
            parameterMatrix[trainSampleIdx].reshape(dimensionNumber, 1) * error[trainSampleIdx]  # (D+1, 1)

        dynamicStep = (iterationNumber - j) / iterationNumber * step                  # reduce step at each iteration

        weightsHistory[:,j] = weights[:,0]                                            # (D+1, 1)

    return weights, weightsHistory


def sampleClassification(inputVector, weights):
    """
    classify this input sample's category based on trained weights data   

    Input:
    - inputVector: (D+1,), the input data's parameter vector
    - weights: (D+1, 1), the final weights values after gradient descent training

    Return:
    - classifyResults: the classify result of this sample based on trained weights data
    """
    probability = sigmoid(np.dot(inputVector, weights))
    if probability >= 0.5:
        classifyResult = 1
    else:
        classifyResult = 0

    return classifyResult

def horseColicPredic(initialTrainStep, iterationNumber):
    """
    use the horseColicTraining.txt data set to train the classifier and use the horseColicTest.txt 
    data set to validate the classifier, then calculate the error rate of the trained classifier in 
    test set.

    Input:
    - initialTrainStep: a scalar which represents the inital step length (value) at gradient descent
    - iterationNumber: a scalar that shows the time of iteration

    Return:
    - testClassifyResults: the classified results of samples in test set
    - testCategory: the actual results of samples in test set
    - errorRate: the error rate of the trained classifier

    """

    trainParaMatrix, trainCategory = loadDataSet('horseColicTraining.txt')
    testParaMatrix, testCategory = loadDataSet('horseColicTest.txt')
    weights,_ = stocGradientDescent(trainParaMatrix, trainCategory, initialTrainStep, iterationNumber)


    error = 0
    testClassifyResults = []
    for i in range(len(testCategory)):
        classifyResult = sampleClassification(testParaMatrix[i], weights)
        testClassifyResults.append(classifyResult)

        if classifyResult != testCategory[i]:
            error += 1

    errorRate = error / len(testCategory)

    return testClassifyResults, testCategory, errorRate
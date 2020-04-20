from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featureLabels = ['no surfacing', 'flippers']
    return dataSet, featureLabels

def createDataSet2():
    dataSet = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    featureLabels = ['no surfacing', 'flippers']
    return dataSet, featureLabels


def ShannonEntropy(dataSet):
    """
    Calculate the Shannon Entropy of this dataSet

    Input:
    - dataSet: a list of N data, every data has D dimensions.

    Return:
    - entropy: The Shannon Entropy of this dataSet, a scaler.

    """
    numOfData = len(dataSet)
    labelCount = {}
    for data in dataSet:
        if data[-1] not in labelCount.keys():
            labelCount[data[-1]] = 1
        else:
            labelCount[data[-1]] += 1

    entropy = 0
    for key in labelCount:
        count = labelCount[key]
        probability = count / numOfData
        entropy -= probability * log(probability, 2)
    return entropy

def splitDataSet(dataSet, splitFeatureIndex, splitFeatureValue):
    """
    split the dataSet by selected feature

    Input:
    - dataSet: a list of N data, every data has D dimensions.
    - splitFeatureIndex: the index of feature that choose to split the dataSet.
    - splitFeatureValue: the particular value of the feature that choose to split the dataSet.

    Return:
    - featureSplitDataSet: a list of M data, every data has D-1 dimensions, exclude the selected feature.
                           these M data are all the data that selected feature is equal particular value.
    
    """
    featureSplitDataSet = []
    for data in dataSet:
        if data[splitFeatureIndex] == splitFeatureValue:
            featureSplitData = data[0:splitFeatureIndex]
            featureSplitData.extend(data[splitFeatureIndex+1:])
            featureSplitDataSet.append(featureSplitData)
    return featureSplitDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    choose the best feature index of the dataSet

    Input:
    - dataSet: a list of N data, every data has D dimensions.

    Return:
    - optimalFeature: The feature that produces the highest information gain comparing to base entropy (entropy of dataSet).
    
    """
    numOfFeatures = len(dataSet[0]) - 1   # because the last column is label, not the feature
    optimalGain = 0
    optimalFeature = -1
    for i in range(numOfFeatures):

        # get all unique values of this feature
        thisFeatureValue = []
        for data in dataSet:
            thisFeatureValue.append(data[i])     
        thisFeatureValue = set(thisFeatureValue)   # only choose the unique values in this feature

        # calculate the information gain of this feature
        GainOfThisFeature = ShannonEntropy(dataSet)
        for value in thisFeatureValue:
            dataSetExcludeThisFeature = splitDataSet(dataSet, i, value)
            GainOfThisFeature -= len(dataSetExcludeThisFeature) / len(dataSet) * ShannonEntropy(dataSetExcludeThisFeature)

        # if the information gain of this feature is greater than previous optimal information gain, choose this as optimal
        if GainOfThisFeature > optimalGain:
            optimalGain = GainOfThisFeature
            optimalFeature = i

    return optimalFeature

def majorityCount(labelList):
    """
    Find the label that appears most of time in the labelList

    Input:
    - labelList: a list of labels

    Return:
    - majorityLabel: the label that appears most of time in the labelList
    
    """
    labelCount = {}
    for label in labelList:
        if label not in labelCount.keys():
            labelCount[label] = 1
        else:
            lebelCount[label] += 1
    sortedLabelCount = sorted(labelCount.items(), key = lambda x: x[1], reverse = True)
    majorityLabel = sortedLabelCount[0][0]

    return majorityLabel

def createTree(dataSet, featureLabels):
    """
    Construct the decision tree

    Input:
    - dataSet: a list of N data, every data has D dimensions
    - featureLabels: a list of D-1 elements, each element is the name of each feature

    Return:
    - myTree: the decision tree that we construct (is a dictionary)
    
    """

    labelList = []
    for data in dataSet:
        labelList.append(data[-1])

    ###################
    # ending criteria #
    ###################
    if labelList.count(labelList[0]) == len(labelList):  # All data are in the same label
        return labelList[0]
    if len(dataSet[0]) == 1:    #  All features are used for classification
        return majorityCount(labelList)


    optimalFeature = chooseBestFeatureToSplit(dataSet)
    optimalFeatureLabel = featureLabels[optimalFeature]
    featureLabels.remove(optimalFeatureLabel)
    subFeatureLabels = featureLabels
    myTree = {optimalFeatureLabel: {}}

    thisFeatureValue = []
    for data in dataSet:
        thisFeatureValue.append(data[optimalFeature])     
    thisFeatureValue = set(thisFeatureValue)   # only choose the unique values in this feature

    for featureValue in thisFeatureValue:
        subDataSet = splitDataSet(dataSet, optimalFeature, featureValue)
        myTree[optimalFeatureLabel][featureValue] = createTree(subDataSet, subFeatureLabels)

    return myTree

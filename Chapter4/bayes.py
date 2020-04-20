import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'], \
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], \
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], \
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],  \
                   ['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], \
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVector = [0, 1, 0, 1, 0, 1]

    return postingList, classVector

def createVocabList(dataSet):
    """
    Create the word list (like a dictionary)

    Input:
    - dataSet: words and vocabulary of the input data

    Return:
    - vocabList: a word list (like a dictionary)

    """
    vocabList = []
    for sentence in dataSet:
        vocabList.extend(sentence)
    vocabList = set(vocabList)
    vocabList = list(vocabList)

    return vocabList

def setOfWords2Vec(vocabList, inputSentence):
    """
    Convert the input sentence to a vector which is consists of 0 and 1,
    the vector length is equal to length of vocabList.

    Input:
    - vocabList: a word list (like a dictionary)
    - inputSentence: the input sentence

    Return:
    - sentenceVector: a vector which is consists of 0 and 1, 
                      the vector length is equal to length of vocabList.

    """
    sentenceVector = [0] * len(vocabList)
    for word in inputSentence:
        if word in vocabList:
            sentenceVector[vocabList.index(word)] = 1
        else:
            print("The word: %s is not vocabulary list" % word)

    return sentenceVector

def bagOfWords2Vec(vocabList, inputSentence):
    """
    Convert the input sentence to a vector, each element indicates the time of the word appears.
    the vector length is equal to length of vocabList.

    Input:
    - vocabList: a word list (like a dictionary)
    - inputSentence: the input sentence

    Return:
    - sentenceVector: a vector, each element indicates the time of the word appears.

    """
    sentenceVector = [0] * len(vocabList)
    for word in inputSentence:
        if word in vocabList:
            sentenceVector[vocabList.index(word)] += 1
        else:
            print("The word: %s is not vocabulary list" % word)

    return sentenceVector

def trainNaiveBayes(trainMatrix, trainCategory):
    """
    train the naive bayes algorithm based on input data and input classify results

    Input:
    - trainMatrix: (N, D), N is the number of training dataSet, 
                   D is the dimension, is equal to length of vocabList.
                   this matrix is consists of 0 and 1.
                   0 represents this word does not appear in this sentance sample, 
                   1 represents this word appears in this sentance sample.
    - trainCategory: (N,), N is the number of training dataSet, this list is consists of 0 and 1.
                     0 represents this sample is non-abusive, 1 represents this sample is abusive.

    Return:
    - probabilityAbusiveVector: (D,), a vector, probability of every word appreance in abusive sentence samples.
    - probabilityNonAbusiveVector: (D,), a vector, probability of every word appreance in non abusive sentence samples.
    - probabilityAbusiveSamples: a scaler, 0 <= probabilityAbusiveSamples <= 1. probability of abusive samples in training samples.

    """  
    trainMatrix = np.array(trainMatrix)
    trainCategory = np.array(trainCategory)

    probabilityAbusiveSamples = sum(trainCategory) / len(trainCategory)  # probability of abusive samples in training samples

    numOfTrainingSamples = len(trainMatrix)      # N
    lengthOfVocab = len(trainMatrix[0])          # D


    ##############################################################################################
    # Not to set all following values to zero is because to avoid one of p(w0|c0), p(w0|c1),     #
    # p(w1|c0), p(w1|c1), ..., p(wn|c0), p(wn|c1) is zero which will make the product to be zero #
    ##############################################################################################
    abusiveVector = np.ones(lengthOfVocab)       # (D,), sum of all abusive sentence vector
    abusiveTotalVocabNum = 2                     # total vocabulary number in abusive sentence samples
    NonAbusiveVector = np.ones(lengthOfVocab)    # (D,), sum of all non abusive sentence vector
    NonAbusiveTotalVocabNum = 2                  # total vocabulary number in non abusive sentence samples

    for i in range(numOfTrainingSamples):
        if trainCategory[i] == 1:
            abusiveVector += trainMatrix[i]
            abusiveTotalVocabNum += sum(trainMatrix[i])
        else:
            NonAbusiveVector += trainMatrix[i]
            NonAbusiveTotalVocabNum += sum(trainMatrix[i])

    probabilityAbusiveVector = abusiveVector / abusiveTotalVocabNum
    probabilityNonAbusiveVector = NonAbusiveVector / NonAbusiveTotalVocabNum

    return probabilityAbusiveVector, probabilityNonAbusiveVector, probabilityAbusiveSamples  
           #  probabilityAbusiveVector is:    p(w0|c1), p(w1|c1), p(w2|c1), ..., p(wn|c1)
           #  probabilityNonAbusiveVector is: p(w0|c0), p(w1|c0), p(w2|c0), ..., p(wn|c0)
           #  probabilityAbusiveSamples is:   p(c1)
           #  in this example, p(c0) = 1 - p(c1)

def naiveBayesClassify(inputSentenceVector, probabilityAbusiveVector, probabilityNonAbusiveVector, probabilityAbusiveSamples):
    """
    Judge this input sentence vector is abusive sentence or not

    Input:
    - inputSentenceVector: (D,), the sentence vector that needs to be classified.
    - probabilityAbusiveVector: (D,), a vector, probability of every word appreance in abusive sentence samples.
    - probabilityNonAbusiveVector: (D,), a vector, probability of every word appreance in non abusive sentence samples.
    - probabilityAbusiveSamples: a scaler, 0 <= probabilityAbusiveSamples <= 1. probability of abusive samples in training samples.

    Return:
    - abusiveSign: 0 or 1, 0 represents that this sentence is not abusive and 1 represents it is abusive.

    """  

    pwc1 = inputSentenceVector * np.log(probabilityAbusiveVector)      # (D,)
    pwc0 = inputSentenceVector * np.log(probabilityNonAbusiveVector)   # (D,)
    pc1w = sum(pwc1) + np.log(probabilityAbusiveSamples)
    pc0w = sum(pwc0) + np.log(1 - probabilityAbusiveSamples)
    # the denominator p(w) for both pc1w and pc0w is the same, so can be neglected.

    if pc1w > pc0w:
        abusiveSign = 1
    else:
        abusiveSign = 0

    return abusiveSign

def textParse(inputString):
    """
    Split the input string into words, keep the words whose lengths are greater than 2.

    Input:
    - inputString: the input string

    Return:
    - l: a list of words, all of these words' lengths are greater than 2.
    """
    import re
    wordList = re.findall(r'\w*', inputString)
    l = []
    for word in wordList:
        if len(word) > 2:
            l.append(word.lower())
    return l

def spamTest():
    """
    load all email samples in ham and spam folder and randomly divide them into two groups:
    training group and test group. Use the training group the train the naive bayes classifier
    and then use the test group to test the classifier.

    Input:
    - None

    Return:
    - classifyResults: the classification results for test group (whether the email is spam or not)
    - testSampleResults: the actual results for test group (whether the email is spam or not)

    """

    import random
    testSampleSize = 10
    AllSampleSize = 50

    wordList = []                 # All words list in ham and spam emails #
    emailWordList = []            # A list of lists, one email's words consist of a list, several ham and   #
                                  # spam emails' word lists combined together to compose the emailWordList  #
    classificationList = []       # A list of classification results, consists of 0 and 1. 0 represents     #
                                  # this sample is ham email, 1 represents this sample is spam email        #                     

    for fileNumber in range(1, int(AllSampleSize/2) + 1):    #  [1, 25] or [1, 26)
        emailString = open('email/ham/%d.txt' % fileNumber).read()
        wordList.extend(textParse(emailString))
        emailWordList.append(textParse(emailString))
        classificationList.append(0)

        emailString = open('email/spam/%d.txt' % fileNumber).read()
        wordList.extend(textParse(emailString))
        emailWordList.append(textParse(emailString))
        classificationList.append(1)

    vocabList = createVocabList(emailWordList)   # create the vocabulary dictionary from all email samples

    # randomly divide the samples into two groups, training group and test group #
    ##############################################################################
    testSampleList = []
    testSampleResults = []
    for i in range(testSampleSize):
        testSampleIdx = random.randint(0, len(emailWordList) - 1)   
        testSampleList.append(emailWordList[testSampleIdx])
        testSampleResults.append(classificationList[testSampleIdx])
        del emailWordList[testSampleIdx]
        del classificationList[testSampleIdx]
    trainSampleList = emailWordList
    trainSampleResults = classificationList
    ##############################################################################

    trainMatrix = []
    for trainSample in trainSampleList:
        trainMatrix.append(bagOfWords2Vec(vocabList, trainSample))
    probabilitySpamVector, probabilityHamVector, probabilitySpamSamples = trainNaiveBayes(trainMatrix, trainSampleResults)

    classifyResults = []
    for testSample in testSampleList:
        testSampleVector = bagOfWords2Vec(vocabList, testSample)
        classifyResult = naiveBayesClassify(testSampleVector, probabilitySpamVector, probabilityHamVector, probabilitySpamSamples)
        classifyResults.append(classifyResult)

    return classifyResults, testSampleResults

def calcMostFreq(vocabList, inputText, topFrequentNumber):
    """
    calculate the most frequent appeared words (for instance, top 100) in this input text,
    the word identification pool is the vocabulary list.

    Input:
    - vocabList: the vocabulary list used to count the word appear time in input text
    - inputText: a list of words that need to be counted
    - topFrequentNumber: the top number of most appeared words, for instance: top30, top50

    Return:
    - mostFreqWordList: the most frequent appeared word list, top number of words which
                        appear in input text with identification pool of vocabulary list.
    - mostFreqWordAppearTimeList: the most frequent appeared word appear time list, same 
                                  length as mostFreqWordList. It shows how many times the
                                  relative word in mostFreqWordList appears in inputText.             
    """    

    wordFrequencyDict = {}   # a list shows how many times of each word (in vocabulary list) appear in input text
    for word in vocabList:
        appearTime = inputText.count(word)
        wordFrequencyDict[word] = appearTime

    valueSorted = sorted(zip(wordFrequencyDict.values(), wordFrequencyDict.keys()), reverse = True)
    mostFreq = valueSorted[0:topFrequentNumber]
    mostFreqWordList = []
    mostFreqWordAppearTimeList = []
    for item in mostFreq:
        mostFreqWordList.append(item[1])
        mostFreqWordAppearTimeList.append(item[0])

    return mostFreqWordList, mostFreqWordAppearTimeList



def feedClassifyTest(feed1, feed0):
    """
    load both feed1 and feed0 text in feed['entries'][entryNumber]['summary'],
    randomly divide the samples into two groups: training group and test group.
    Then remove the top frequent appeared words in vocabulary list, use this 
    modified vocabulary list and training group to train the naive bayes classifier.
    At last use the test group the test the classifier.  

    Input:
    - feed1: feed category 1
    - feed0: feed category 0

    Return:
    - classifyResults: the classification results for test group (whether the feed is in feed1 or feed0)
    - testSampleResults: the actual results for test group (whether the feed is in feed1 or feed0)

    """

    import random 
    trainSampleProportion = 0.8
    removeTopWordsNumber = 30

    wordList = []                 
    RSSWordList = []
    classificationList = []                    

    minLength = min(len(feed1['entries']), len(feed0['entries']))
    for RSSEntryNumber in range(minLength):   
        wordList.extend(textParse(feed0['entries'][RSSEntryNumber]['summary']))
        RSSWordList.append(textParse(feed0['entries'][RSSEntryNumber]['summary']))
        classificationList.append(0)

        wordList.extend(textParse(feed1['entries'][RSSEntryNumber]['summary']))
        RSSWordList.append(textParse(feed1['entries'][RSSEntryNumber]['summary']))
        classificationList.append(1)


    vocabList = createVocabList(RSSWordList) 
    # remove the most frequent appeared words in vocabulary list   #########
    ########################################################################
    topWords,_ = calcMostFreq(vocabList, wordList, removeTopWordsNumber)
    for topWord in topWords:
        vocabList.remove(topWord)
    ########################################################################

    # randomly divide the samples into two groups, training group and test group #
    ##############################################################################
    testSampleList = []
    testSampleResults = []
    testSampleSize = int((1 - trainSampleProportion) * 2 * minLength)

    for i in range(testSampleSize):
        testSampleIdx = random.randint(0, len(RSSWordList) - 1)   
        testSampleList.append(RSSWordList[testSampleIdx])
        testSampleResults.append(classificationList[testSampleIdx])
        del RSSWordList[testSampleIdx]
        del classificationList[testSampleIdx]
    trainSampleList = RSSWordList
    trainSampleResults = classificationList
    ##############################################################################

    trainMatrix = []
    for trainSample in trainSampleList:
        trainMatrix.append(bagOfWords2Vec(vocabList, trainSample))
    probability1Vector, probability0Vector, probability1Samples = trainNaiveBayes(trainMatrix, trainSampleResults)

    classifyResults = []
    for testSample in testSampleList:
        testSampleVector = bagOfWords2Vec(vocabList, testSample)
        classifyResult = naiveBayesClassify(testSampleVector, probability1Vector, probability0Vector, probability1Samples)
        classifyResults.append(classifyResult)

    return classifyResults, testSampleResults

def getTopWords(feed1, feed0, removeTopWordsNumber, appearTimeThreshold):
    """
    load both feed1 and feed0 text in feed['entries'][entryNumber]['summary'],
    Then remove the top frequent appeared words in the two feeds. At last keep
    the words with appear times greater than appear time threshold in both feed1
    and feed0. 

    Input:
    - feed1: feed category 1
    - feed0: feed category 0
    - removeTopWordsNumber: how many most frequent words we want to remove from the vocabList
    - oppearTimeThreshold: after remove the top frequent words, for appear time of remaining 
                           words, if its appear time is greater than this value, keep this word.
                           Otherwise discard this word.

    Return:
    - topWords0: a word list, these words (from feed0) appeared times are greater than 
                 oppearTimeThreshold after remove the top frequent words
    - topWords1: a word list, these words (from feed1) appeared times are greater than 
                 oppearTimeThreshold after remove the top frequent words
    """

    wordList0 = []                 
    RSSWordList0 = [] 
    wordList1 = []                 
    RSSWordList1 = []            

    for RSSEntryNumber in range(len(feed0['entries'])):
        wordList0.extend(textParse(feed0['entries'][RSSEntryNumber]['summary']))
        RSSWordList0.append(textParse(feed0['entries'][RSSEntryNumber]['summary']))
    for RSSEntryNumber in range(len(feed1['entries'])):
        wordList1.extend(textParse(feed1['entries'][RSSEntryNumber]['summary']))
        RSSWordList1.append(textParse(feed1['entries'][RSSEntryNumber]['summary']))

    # load all words and their appear times in topWords0/topWords1 and topWordsAppearTime0/topWordsAppearTime1 #
    ############################################################################################################
    vocabList0 = createVocabList(RSSWordList0)
    topWords0, topWordsAppearTime0 = calcMostFreq(vocabList0, wordList0, len(vocabList0))
    vocabList1 = createVocabList(RSSWordList1) 
    topWords1, topWordsAppearTime1 = calcMostFreq(vocabList1, wordList1, len(vocabList1))
    ############################################################################################################

    # remove the most frequent appeared words and their appear times from ####################
    # topWords0/topWords1 and topWordsAppearTime0/topWordsAppearTime1     ####################
    ##########################################################################################
    topWords0 = topWords0[removeTopWordsNumber:len(topWords0)]
    topWordsAppearTime0 = topWordsAppearTime0[removeTopWordsNumber:len(topWordsAppearTime0)]
    topWords1 = topWords1[removeTopWordsNumber:len(topWords1)]
    topWordsAppearTime1 = topWordsAppearTime1[removeTopWordsNumber:len(topWordsAppearTime1)]
    ##########################################################################################

    # to see whether the remaining words' appear times are greater than threshold value or not. #
    # If yes, load this word into topWords0 or topWords1. If no, dicard this word.              #
    #############################################################################################
    for appearTime in topWordsAppearTime0:
        if appearTime < appearTimeThreshold:
            idx = topWordsAppearTime0.index(appearTime)
            break
    topWords0 = topWords0[0:idx]

    for appearTime in topWordsAppearTime1:
        if appearTime < appearTimeThreshold:
            idx = topWordsAppearTime1.index(appearTime)
            break    
    topWords1 = topWords1[0:idx]
    ##############################################################################################
    return topWords0, topWords1

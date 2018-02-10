from numpy import *
import math


class NB:
    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
        return postingList, classVec

    def createVocabList(self, dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return list(vocabSet)

    def setOfWords2Vec(self, vocabList, inputSet):
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print('the word: %s is not in my vocablary!' % word)
        return returnVec

    def trainNBO(self, trainMatrix, trainCategory):
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory) / float(numTrainDocs)
        p0Num = zeros(numWords)
        p1Num = zeros(numWords)
        p0Demon = 0.0
        p1Demon = 0.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Demon += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Demon += sum(trainMatrix[i])
        # p1Vect=[math.log(x) for x in p1Num/p1Demon]
        # p0Vect=[math.log(x) for x in p0Num/p0Demon]
        p1Vect = p1Num / p1Demon
        p0Vect = p0Num / p0Demon
        return p0Vect, p1Vect, pAbusive

    def classifyNB(self, vecClassify, p0Vec, p1Vec, pClass1):
        p1 = sum(vecClassify * p1Vec) + math.log(pClass1)
        p0 = sum(vecClassify * p0Vec) + math.log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

    def testingNB(self):
        listOPosts, listClasses = self.loadDataSet()
        myVocabList = self.createVocabList(listOPosts)
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(self.setOfWords2Vec(myVocabList, postinDoc))
        p0V, p1V, pAb = self.trainNBO(array(trainMat), array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(self.setOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAb))
        testEntry = ['stupid', 'garbage']
        thisDoc = array(self.setOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    print('TEST')
    nb = NB()
    nb.testingNB()

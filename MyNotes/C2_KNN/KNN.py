from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


class KNN:
    def createDataSet(self):
        group = array([[1.0, 1.1],
                       [1.0, 1.0],
                       [0, 0],
                       [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def classify0(self, inx, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = tile(inx, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()  # 从小到大排序返回索引
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def file2matrix(self, filename):
        fr = open(filename)
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)
        returnMat = zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector

    def autoNorm(self, dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m, 1))
        normDataSet = normDataSet / tile(ranges, (m, 1))
        return normDataSet, ranges, minVals

    def datingClassTest(self):
        hoRatio = 0.1
        datingDataMat, datingLabels = self.file2matrix('data/datingTestSet2.txt')
        normMat, ranges, minVals = self.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVec = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVec):
            classifierResult = self.classify0(normMat[i, :], normMat[numTestVec:m, :], datingLabels[numTestVec:m], 3)
            print('the classifier came back with: %d,the real answer is: %d'
                  % (classifierResult, datingLabels[i]))
            if classifierResult != datingLabels[i]:
                errorCount += 1.0
        print('the total error rate is: %f' % (errorCount / float(numTestVec)))


if __name__ == '__main__':
    print('TEST')
    knn = KNN()
    l = knn.datingClassTest()
    # #散点图
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
    #            15.0*array(labelVector),15.0*array(labelVector))
    # plt.show()

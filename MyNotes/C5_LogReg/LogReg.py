from numpy import *


class logReg:
    def loadDataSet(self):
        dataMat = []
        labelMat = []
        fr = open('testSet.txt')
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat

    def sigmoid(self, inX):
        return 1.0 / (1 + exp(-inX))

    def gradAscent(self, dataMatIn, classLabels):
        dataMatrix = mat(dataMatIn)
        labelMat = mat(classLabels).transpose()
        m, n = shape(dataMatrix)
        alpha = 0.001
        maxCycles = 500
        weights = ones((n, 1))
        for k in range(maxCycles):
            h = self.sigmoid(dataMatrix * weights)
            error = labelMat - h
            weights = weights + alpha * dataMatrix.transpose() * error
        return weights

    def stoGradAscent1(self, dataMatrix, classLabels):
        m, n = shape(dataMatrix)
        alpha = 0.01
        weights = ones(n)
        for i in range(m):
            h = self.sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        return weights

    def stoGradAscent2(self, dataMatrix, classLabels, numIter=150):
        m, n = shape(dataMatrix)
        weights = ones(n)
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.01  # alpha每次迭代都调整，不断减小但是避免严格下降
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = self.sigmoid(sum(dataMatrix[randIndex] * weights))  # 随机选取样本
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                del (dataIndex[randIndex])
        return weights


if __name__ == '__main__':
    logR = logReg()
    dataArr, labelMat = logR.loadDataSet()
    a = logR.stoGradAscent2(array(dataArr), labelMat)
    print(a)

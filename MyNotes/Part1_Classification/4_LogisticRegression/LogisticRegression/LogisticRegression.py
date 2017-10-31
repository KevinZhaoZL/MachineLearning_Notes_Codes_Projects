import math
import numpy as np
from numpy import *
import random
"""
逐行读取文本文件中的数据，每行两个值再加上第三个对应的类别标签
"""
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    i=1
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
"""
Sigmoid函数
"""
def sigmoids(inX):
    for line in inX:
        line[0][0]=1.0/(1+math.exp(-line[0][0]))
    return inX
def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))

"""
梯度上升算法
输入：特征值和标签列表
输出：回归系数矩阵
"""
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoids(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
"""
随机梯度上升算法
输入：特征值和标签列表
输出：回归系数矩阵
"""
def stocGradAscent0(dataMatrix,classLabels):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        d=dataMatrix[i]
        weights[0] = weights[0] + alpha * error * float(dataMatrix[i][0])
        weights[1] = weights[1] + alpha * error * float(dataMatrix[i][1])
        weights[2] = weights[2] + alpha * error * float(dataMatrix[i][2])
    return weights
"""
改进的随机梯度上升算法
"""
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights[0] = weights[0] + alpha * error * float(dataMatrix[randIndex][0])
            weights[1] = weights[1] + alpha * error * float(dataMatrix[randIndex][1])
            weights[2] = weights[2] + alpha * error * float(dataMatrix[randIndex][2])
    return weights

"""
预测病马
"""
#以回归系数和特征向量作为输入计算对应的Sigmoid值，并条件返回0 1
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
#训练并测试
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    #训练分类器
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[i]))
    trainingWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    #测试分类器
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
#调用10次上面的方法求均值
def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))

"主模块用于测试"
if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    #print(gradAscent(dataArr,labelMat))
    #print(stocGradAscent1(dataArr,labelMat))
    multiTest()























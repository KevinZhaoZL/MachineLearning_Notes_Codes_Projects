from math import log
import operator
import treePlotter

"""
计算香农熵
输入：特征值和标签组成的矩阵
输出：该数据集的香农熵
"""
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    #统计各标签及其所占的次数
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    #计算香农熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

"""
按照给定的特征划分数据集
输入：带划分数据集，特征，特征返回值
输出：符合给定条件的数据记录
"""
def splitDataSet(dataSet,axis,value):
    #创建一个列表用于存放特征值符合给定条件的数据记录
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            temp=featVec[:axis]
            temp.extend(featVec[axis+1:])
            retDataSet.append(temp)
    return retDataSet

"""
从数据集中选择最佳划分特征
输入：数据集矩阵
输出：最佳特征
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        #取得第i个特征的所有取值
        featList=[example[i] for example in dataSet]
        #得到第i个特征的可能的所有不重复取值
        uniqueVals=set(featList)
        #划分后的信息熵
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            #将所有划分的子集的信息熵加起来
            newEntropy+=prob*calcShannonEnt(subDataSet)
        #信息增益=旧熵-新熵   {熵代表数据集的无序程度，信息增益就是熵的减小值，变整齐了多少..}
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

"""
多数表决
输入：特征值出现的列表
输出：出现次数逆序的特征值列表
"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

"""
创建树
输入：数据集和标签列表
输出：用字典表示的树
"""
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVlas=set(featValues)
    for value in uniqueVlas:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

"""
使用pickle模块储存决策树
"""
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)
#主方法用于测试
if __name__ == "__main__":
    myDat=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']]
    labels=['no surfacing','flippers']
    #print(calcShannonEnt(myDat))
    #print(splitDataSet(myDat,0,1))
    #print(chooseBestFeatureToSplit(myDat))
    #myTree=createTree(myDat,labels)
    myTree={'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    treePlotter.createPlot(myTree)























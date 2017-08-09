import operator
import numpy as np

"""
kNN核心分类器
输入
inX：用于分类的输入向量
dataSet：训练集
labels：类别标签向量
k：选择邻居的个数
"""
def classify0(inX,dataSet,labels,k):
    #得到训练集的第一纬度的长度(有几行)
    dataSetSize=dataSet.shape[0]
    #得到特征值之间的差，计算欧式距离
    #将输入向量沿行扩展得到差矩阵
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    #差矩阵各个元素乘二次方
    sqDiffMat=diffMat**2
    #求每一行元素的和并开方得到欧式距离矩阵
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #将欧氏距离排序，argsort返回数组值从小到大的索引值
    sortedDistIndicies=distances.argsort()
    #得到前k个中出现次数最多的类别标签
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndicies[i]]
        #给不同的voteLabel计数
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    #sorted第一个参数是一个迭代器，对于字典排序，返回结果是一个元素为元组的列表
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    #返回出现次数最多的voteLabel
    return sortedClassCount[0][0]
"""
文本数据转换为矩阵数据
输入：文件路径
输出：训练样本矩阵和类标签向量
"""
def file2matrix(filename):
    fr=open(filename)
    #得到样本数据的行数
    arrayOfLines=fr.readlines()
    numberOfLines=len(arrayOfLines)
    #初始化一个0矩阵用于存放样本数据
    returnMat=np.zeros((numberOfLines,3))
    #定义一个向量用于存放标签
    classLabelVector=[]
    index=0
    #解析数据到矩阵和向量中去
    for line in arrayOfLines:
        #删除开头和结尾的空白符(包括回车，回车换行和制表符)
        line=line.strip()
        #根据制表符切割每一行的数据
        listFromLine=line.split('\t')
        #将列表写到矩阵中
        returnMat[index,:]=listFromLine[0:3]
        #每一行最后一个是标签
        classLabelVector.append(int(listFromLine[-1]))
        #到下一行
        index+=1
    return returnMat,classLabelVector

"""
归一化特征值
输入：原始特征矩阵
输出：元素值属于[0,1]区间的新的特征矩阵
"""
def autoNorm(dataSet):
    #从列选取最值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    #初始化一个0矩阵
    normDataSet=np.zeros(np.shape(dataSet))
    #列数
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

"""
测试代码
"""
def dataingTest():
    hoRatio=0.10
    datingSet,datainglabels=file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals=autoNorm((datingSet))
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datainglabels[numTestVecs:m],3)
        print("result is %d, answer is %d"%(classifierResult,datainglabels[i]))
        if(classifierResult!=datainglabels[i]):
            errorCount+=1.0
    print("error rate is %f"%(errorCount/float(numTestVecs)))

dataingTest()









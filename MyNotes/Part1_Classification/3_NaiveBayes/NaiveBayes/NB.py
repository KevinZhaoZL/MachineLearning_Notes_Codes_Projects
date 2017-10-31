from numpy import *

"""
方法返回经过切分且去除标点的词条向量和每个词条向量人工标注的侮辱性or非侮辱性
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
"""
方法统计所有在文档中出现的单词
输入：词条向量矩阵
输出：列表形式的出现词汇表
"""
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

"""
方法将单个词条向量转换成与词汇表等长的1 0向量(词集模型)
输入：词汇表和词条向量
输出：词汇表等长的向量（0/1）
"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("word %s is not in the vocabulary!" % word)
    return  returnVec

"""
朴素贝叶斯词袋模型
"""
def bagofWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

"""
朴素贝叶斯分类器训练函数
输入：文档矩阵，每篇文档类别标签所构成的向量
输出：在给定类别下各个单词出现的概率，文档属于侮辱类的概率
"""
def trainNBO_0(trainMatrix,trainCategory):
    #有多少篇文档
    numTrainDocs=len(trainMatrix)
    #每个文档有多少个词,实际上也是词集的长度
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

"""
修改后的朴素贝叶斯分类器训练函数
"""
def trainNBO_1(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

"""
朴素贝叶斯分类函数
输入：要分类的向量，使用trainNBO_1计算得到的概率
输出：类别
"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

"""
封装所有的操作，使其简单
"""
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    poV,p1V,pAb=trainNBO_1(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classified as: ",classifyNB(thisDoc,poV,p1V,pAb))
    testEntry=['stupid','gardage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"classified as: ",classifyNB(thisDoc,poV,p1V,pAb))

"""
将邮件中乱七八糟的内容切分转换成词的列表
"""
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

"""
垃圾邮件分类器的训练与测试
"""
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #导入文件下的文本文件，解析使之成为词列表
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    #留存交叉验证，选择十个文件作为测试集，并删除之，其余构成训练集
    vocabList=createVocabList(docList)
    trainingSet=range(50)
    testSet=[]
    j=50
    for i in range(10):
        randIndex=int(random.uniform(0,j))
        testSet.append(trainingSet[randIndex])
        trainingSet=list(trainingSet)
        del(trainingSet[randIndex])
        j=j-1
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNBO_1(array(trainMat),array(trainClasses))
    errorCount=0
    #对测试集分类并计算错误率
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("the error rate is : ",float(errorCount)/len(testSet))


if __name__ == "__main__":
    #testingNB()
    spamTest()











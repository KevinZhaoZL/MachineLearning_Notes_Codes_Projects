from numpy import *

def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        ai=H
    if L>aj:
        aj=L
    return aj

"""
简化版SMO算法
输入：数据集，类别标签，常数C，容错率，取消前最大的循环次数
输出：f(x)中的b值和拉格朗日乘子alpha
"""
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    #数据，临时参数的设置
    dataMatrix=mat(dataMatIn);labelMat=mat(classLabels).transpose()
    b=0;m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    iter=0
    #循环过程
    while(iter<maxIter):
        #记录alpha是否已经进行优化
        alphaPairsChanged=0
        for i in range(m):
            #预测的类别
            fXi=float(multiply(alphas,labelMat).T*\
                      (dataMatrix*dataMatrix[i:].T))+b
            #预测与实际类别的误差(正误差和负误差)
            Ei=fXi-float(labelMat[i])
            #检查alpha值：保证其不为0或者C；满足已设定的一些条件
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or\
                    ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                #随机函数选择第二个alpha值，并计算误差
                j=selectJrand(i,m)
                fXj=float(multiply(alphas,labelMat).T*\
                          (dataMatrix*dataMatrix[j:].T))+b
                Ej=fXj-float(labelMat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                #计算L和H，用于调整第二个alpha值到0~C之间
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                #相等->不做改变
                if L==H:
                    print("L=H")
                    continue
                #计算alpha的最优修改量，eta = K11+K22-2*K12,也是f(x)的二阶导数
                eta=2.0*dataMatrix[i:]*dataMatrix[j:].T-\
                    dataMatrix[i,:]*dataMatrix[i:].T-\
                    dataMatrix[j,:]*dataMatrix[j:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                #修改这一对alpha，值相同，方向相反
                alphas[i]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j].H,L)
                #检查alpha[j]是否轻微改变
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*\
                           (alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*\
                    dataMatrix[i,:]*dataMatrix[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJold)*\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b=b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print("iter:%d i: %d,pairs changed %d" %(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):iter+=1
        else:iter=0
        print("iteration numbe: %d" %iter)
    return b,alphas

#一个数据结构
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        #误差缓存
        self.eCache=mat(zeros((self.m,2)))

def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*\
              (oS.X*os.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return  Ek

"""
内循环中的启发式方法
"""
def selectJ(i,oS,Ei):
    maxk=-1
    maxDeltaE=0
    Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k
                maxDeltaE=deltaE
                Ej=Ek
        return  maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
        return j,Ej
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

"""
基于alpha得到超平面并输出分类
"""
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

"主模块用于测试"
if __name__=='__main__':
    print("the end")
    dataMat,labelMat=loadDataSet('testSet.txt')
    b,alpha=smoSimple(dataMat,labelMat,0.6,0.001,60)
    print(b)
    print(alpha)














































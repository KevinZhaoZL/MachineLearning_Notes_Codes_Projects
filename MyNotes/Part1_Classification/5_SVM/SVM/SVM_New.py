from numpy import *
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
        j=selectJrand





























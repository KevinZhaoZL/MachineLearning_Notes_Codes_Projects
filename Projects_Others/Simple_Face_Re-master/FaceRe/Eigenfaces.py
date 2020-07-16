# -*-coding:utf-8 -*-
from numpy import *
import numpy as np
import sys,os
import copy
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt

class Eigenfaces(object):
    def __init__(self):
        self.eps=1.0e-16
        self.X=[]
        self.y=[]
        self.Mat=[]
        self.eig_v=0
        self.eig_vect=0
        self.mu=0
        self.projections=[]
        self.dist_metric=0

    def loadimgs(self,path):
        classlabel=0
        for dirname,dirnames,filenames in os.walk(path):
            for subdirname in dirnames:
                sub_path=os.path.join(dirname,subdirname)
                for filename in os.listdir(sub_path):
                    a=os.path.join(sub_path+'/',filename)
                    im=Image.open(a)
                    im=im.convert("L")
                    self.X.append(np.asarray(im,dtype=np.float64))
                    self.y.append(classlabel)
                classlabel+=1
    #图片转换为一个行向量构成的矩阵集合
    def genRowMatrix(self):
        self.Mat=np.empty((0,self.X[0].size),dtype=self.X[0].dtype)
        for row in self.X:
            self.Mat=np.vstack((self.Mat,np.asarray(row).reshape(1,-1)))
    #计算特征脸
    def PCA(self,k=0):
        self.genRowMatrix()
        [n,d]=shape(self.Mat)
        #if(pc_num<=0) or (k>n):
        if k>n:
            k=n
        self.mu=self.Mat.mean(axis=0)
        self.Mat-=self.mu
        if n>d:
            xTx=np.dot(self.Mat.T,self.Mat)
            [self.eig_v,self.eig_vect]=linalg.eigh(xTx)
        else:
            xTx=np.dot(self.Mat,self.Mat.T)
            [self.eig_v,self.eig_vect]=linalg.eigh(xTx)
        self.eig_vect=np.dot(self.Mat.T,self.eig_vect)
        for i in range(n):
            self.eig_vect[:,i]=self.eig_vect[:,i]/linalg.norm(self.eig_vect[:,i])
        idx=np.argsort(-self.eig_v)
        self.eig_v=self.eig_v[idx]
        self.eig_vect=self.eig_vect[:,idx]
        self.eig_v=self.eig_v[0:k].copy()
        self.eig_vect=self.eig_vect[:,0:k].copy()
    def compute(self):
        self.PCA()
        for xi in self.X:
            self.projections.append(self.project(xi.reshape(1,-1)))
    def disEclud(self,vecA,vecB):
        return linalg.norm(vecA-vecB)+self.eps
    def cosSim(self,vecA,vecB):
        return (dot(vecA,vecB.T)/((linalg.norm(vecA)*linalg.norm(vecB))+self.eps))[0,0]
    def project(self,XI):
        if self.mu is None:return np.dot(XI,self.eig_vect)
        return np.dot(XI-self.mu,self.eig_vect)
    def subplot(self,title,images):
        fig=plt.figure()
        fig.text(.5,.95,title,horizontalalignment='center')
        for i in range(len(images)):
            ax0=fig.add_subplot(5,6,(i+1))
            plt.imshow(asarray(images[i]),cmap="gray")
            plt.xticks([]),plt.yticks([])
        plt.show()
    def predict(self,XI):
        minDist=np.finfo('float').max
        minClass=-1
        a=XI.reshape(1,-1)
        Q=self.project(a)
        #Q=self.project(XI.reshape(1,-1))
        for i in range(len(self.projections)):
            dist=self.dist_metric(self.projections[i],Q)
            if dist<minDist:
                minDist=dist
                minClass=self.y[i]
        return minClass

if __name__=='__main__':
    ef=Eigenfaces()
    ef.dist_metric=ef.disEclud
    ef.loadimgs("att_faces/")
    ef.compute()
    # E=[]
    # X=mat(zeros((10,10304)))
    # for i in range(30):
    #     X=ef.Mat[i*10:(i+1)*10,:].copy()
    #     X=X.mean(axis=0)
    #     imgs=X.reshape(112,92)
    #     E.append(imgs)
    # ef.subplot(title="AT&T Eigen Facedatabase",images=E)
    testImg=ef.X[40]
    print("fact",ef.y[40],"->","predict=",ef.predict(testImg))

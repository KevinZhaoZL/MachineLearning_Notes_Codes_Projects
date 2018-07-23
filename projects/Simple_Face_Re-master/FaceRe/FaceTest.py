import os

from PIL import Image
from __builtin__ import xrange
from numpy import *
import numpy as np
from cv2 import *
import cv2

class FaceRe:
    #从窗口显示一幅图片
    def showPicWin(self):
        win_name='myPic'
        #cv2.WINDOW_NORMAL:可以手动调整窗口大小
        cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
        img=cv2.imread('myPic.jpg',1)#0：黑白图片  1：原色图片
        cv2.imshow(win_name,img)
        cv2.waitKey(0)
    #拍照
    def takePics(self):
        cam=VideoCapture(0)
        for x in range(0,1):
            s,img=cam.read()
            if s:
                imwrite("o-"+str(x)+".jpg",img)
    #Haar cascade 实现人脸识别
    def FaceReHaarcascade(self):
        face_cascade=cv2.CascadeClassifier('C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml')
        img=cv2.imread('myPic.JPG')
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #识别输入图片中的人脸对象，返回对象的矩形尺寸
        #函数原型detectMultiScale(gray，1.2,3，CV_HAAR_SCALE_IMAGE,Size(30,30))
        #gray需要识别的图片
        #1.2：每次图像尺寸减小的比例
        #3：表示每一个目标至少要被检测到4次才算是真的目标
        #CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size（30,30）为目标的最小最大尺寸
        #faces:表示检测到的人脸目标序列
        faces=face_cascade.detectMultiScale(gray,1.2,3)
        for (x,y,w,h) in faces:
            img2=cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=img[y:y+h,x:x+w]
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("myPic.head.jpg".img)


if __name__=='__main__':
    FR=FaceRe()
    FR.takePics()
    FR.FaceReHaarcascade()




































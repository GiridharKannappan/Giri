# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:57:07 2019

@author: 
"""
import os
import cv2
#coloured image
img_path=os.path.join(os.getcwd(),'face1.jpg')
img=cv2.imread(img_path,1)
img4=cv2.imread("G:\prog\face1.jpg",1)
img2=cv2.imread('‪G:\prog\719623- military bnigade.jpg',1)
print(img)
#type
type(img4)
#shape
img.shape
img4.shape
#to display image
cv2.imshow('legend',img)
cv2.waitKey(0)
#cv2.waitKey(2000)
cv2.destroyAllWindows()
#Resizing
resize=cv2.resize(img,(600,600))
cv2.imshow('legend',resize_imghalf)
#cv2.waitKey(0)
cv2.waitKey(2000)
cv2.destroyAllWindows()
#to reduce the img size into half
resize_imghalf=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
#to increse img size twice
resize_imgtwice=cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))
cv2.imshow('legend',resize_imgtwice)
cv2.waitKey(0)
#cv2.waitKey(2000)
cv2.destroyAllWindows()
# Face detection
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"harrcascade_eye.xml")
img=cv2.imread(img_path)
img.shape
grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# search for cordinates of the image
face=face_cascade.detectMultiScale(grey_img,scaleFactor=1.05,minNeighbors=5)
#print(type(face))
#print(face)
for x,y,w,h in face:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow('Face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#CAPTURING VIDEOS 
import time,cv2
video=cv2.VideoCapture(0)
while True:
check,frame = video.read()
print(check)
print(frame)
time.sleep(3)
cv2.imshow('Capture',frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
# To capture the vidoes
video2=cv2.VideoCapture(0)
a=0
while True:
    a+=1
    check,frame= video2.read()
    print(frame)
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Capturing',grey)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
print(a)
video.release()
cv2.destroyAllWindows()
#MOTION DETECTOR
#Threshold diff >30 while or <30black
import datetime
import pandas as pd
import cv2
status_list=[None,None]
times=[]
first_frame=None
df=pd.DataFrame(columns=['Start','End'])
video3=cv2.VideoCapture(0)
while True:
    check,frame= video3.read()
    status=0
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grey=cv2.GaussianBlur(grey,(21,21),0)
    if first_frame is None:
        first_frame = grey 
        continue
    diff=cv2.absdiff(first_frame,grey)#calculate diff b/w frames
    thresh=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh=cv2.dilate(diff,None,iterations=2)
    
    (image, cnts, hierarchy) = 
cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE) 
    
#contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status=1
        (x,y,w,h)=cv2.boundinRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_list=append(status)
    status_list=status_list[-2:]
    #store the data when object start and close
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    #store the data when object reoccurs
    if status_list[-1]==1 and status_list[-2]==1:
        times.append(datetime.now())
    print(status_list)
    print(times)
    cv2.imshow('frame',frame)
    cv2.imshow('Grey',Grey)
    cv2.imshow('Difference',diff)
    cv2.imshow('Threshold',thresh)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
for i in range(0,len(times),2):
        
df=df.append({start:times[i],"End":times[i+1]},ignore_index=True)    
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows()


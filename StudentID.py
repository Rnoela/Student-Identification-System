from turtle import width
import cv2
import numpy as npy
import  face_recognition as face_rec
import os
from datetime import datetime

#defining function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = 'Student_images'
studentImg = []
studentName = []
uonList = os.listdir(path)
#print(uonList)

#Load or read all names of registered UoN students and store in studentName

for cl in uonList : 
    curImg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curImg)
    studentName.append(os.path.splitext(cl)[0])
#print(studentName)
# encode all student images/faces and store in encodingList

def findEncoding(images) :
    encodingList = []

    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_rec.face_encodings(img)[0]
        encodingList.append(encodeImg)
    return encodingList 

    #create attendance List in Excelsheet
def MarkAttendance(name):
    with open('classAttendanceList.csv', 'r+') as f:
        myDataList= f.readlines()
        nameList= []
        for Line in myDataList :
            entry = Line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')




encodeList = findEncoding(studentImg)

videoImg = cv2.VideoCapture(0)

while True :
    success, frame = videoImg.read()
    #smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    #frames = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    facesInFrame = face_rec.face_locations(frame)
    encodeFacesInFrame = face_rec.face_encodings(frame, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(encodeList, encodeFace)
        faceDist = face_rec.face_distance(encodeList, encodeFace)
        print(faceDist)
        matchIndex = npy.argmin(faceDist)

        if matches[matchIndex] :
            name =studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            #y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
           # cv2.rectangle(frame, (x1, y1-7), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)
        
    cv2.imshow('video of Student in Attendance', frame)
    cv2.waitKey(1)
        

        





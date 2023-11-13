import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('video_gkt.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

area1=[(380,380),(380,395),(650,395),(650,380)]
area2=[(380,360),(380,375),(650,375),(650,360)]

people_enter={}
counter1=[]

people_exit={}
counter2=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]         
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if results>=0:
            people_exit[id]=(x4,y4)
        if id in people_exit:
            results1=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
            if results1>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                if counter2.count(id)==0:
                    counter2.append(id)

        results2=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
        if results2>=0:
            people_enter[id]=(x4,y4)
        if id in people_enter:
            results3=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
            if results3>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                if counter1.count(id)==0:
                    counter1.append(id)

        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1)    
    er=(len(counter1))
    et=(len(counter2))
    cvzone.putTextRect(frame,f'enter : {er}',(50,50),2,2,(40,0,0),(255, 228, 225))
    cvzone.putTextRect(frame,f'exit : {et}',(50,90),2,2,(0,0,40),(255, 228, 225))
    cvzone.putTextRect(frame,'SYSTEM PENGHITUNG MAHASISWA MASUK GKT',(40,480),1,2,(40,0,0),(255, 228, 225), cv2.FONT_HERSHEY_TRIPLEX)
    cvzone.putTextRect(frame,'Di Program oleh: Desy Portuna',(40,430),1,1,(0,0,40),(255, 228, 225))

 
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()


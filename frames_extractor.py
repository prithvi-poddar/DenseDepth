import cv2
import numpy as np

cap = cv2.VideoCapture("videos/road1.mp4")
count = 0
while True:
    count+=1
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480)) 
    if ret:
        cv2.imwrite("extracted_frames/frame%d.jpg" % count, frame)
        
        
    else:
        break
   
cv2.destroyAllWindows()
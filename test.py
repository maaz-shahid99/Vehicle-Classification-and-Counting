import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap= cv2.VideoCapture("highway.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


while True: 

  _, frame = cap.read()
  height, width, _ = frame.shape

  #print(height, width)

  #Region of interest
  roi = frame[340:720 , 500:800]
  # 1.Object Detection
  
  mask = object_detector.apply(roi)
  _, mask = cv2.threshold(mask, 254,255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  detections=[]

  for cnt in contours:

    # Calculate area and remove contours
    
    area = cv2.contourArea(cnt)
    if area > 100:
      
      #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
      x,y,w,h = cv2.boundingRect(cnt)
      cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 2)

      detections.append([x,y,w,h])
  
  # 2. Object Tracking
  boxes_ids = tracker.update(detections)
  print(boxes_ids)

  cv2.imshow("roi", roi)
  #cv2.imshow("mask", mask)


  if cv2.waitKey(10) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np

from Detector import bugs_detector
from Tracker import bugs_tracker

cap = cv.VideoCapture("data/TrackingBugs.mp4")

detector = bugs_detector()
tracker = bugs_tracker()

skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
while(cap):
    
    ret, frame = cap.read()
    
    if skip_frame_count < 15:
        skip_frame_count+=1
        continue
    
    # detects bugs
    detections = detector.detect(frame)
    
    if len(detections)>0:
        
        tracker.update(detections)
        
        for i in range(len(tracker.tracks)):
            if len(tracker.tracks[i].trace) >1 :
                for j in range(len(tracker.tracks[i].trace)-1):
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    clr = tracker.tracks[i].trackId % 9
                    cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), track_colors[clr], 2)
        
        # Display the resulting tracking frame
        cv.imshow('Tracking', frame)
    
    cv.waitKey(50)
    

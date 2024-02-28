import cv2 as cv
import numpy as np


class bugs_detector:
    
    def __init__(self) -> None:
        self.segThres = 150
        self.fgbg = cv.createBackgroundSubtractorMOG2()
    
    def detect(self, frame):
        
        im_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        # Perform Background Subtraction
        fgmask = self.fgbg.apply(im_gray)
        
        # Detect edges
        edges = cv.Canny(fgmask, 50, 190, 3)
        
        # Retain only edges within the threshold
        ret, thresh = cv.threshold(edges, 127, 255, 0)
        
        # Find contours
        contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        centers = [] 
        blob_radius_thresh = 8
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv.minEnclosingCircle(cnt)
                centeroid = (int(x), int(y))
                radius = int(radius)
                if (radius > blob_radius_thresh):
                    cv.circle(frame, centeroid, radius, (0, 255, 0), 2)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass
        
        
        
        return centers
        
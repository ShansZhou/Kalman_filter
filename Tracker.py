import numpy as np
import scipy.optimize as sciOpt
from Kalman_filter import KalmanFilter



class Track:
    def __init__(self, prediction, track_id) -> None:
        
        self.trackId = track_id
        self.KF = KalmanFilter()
        self.prediction = np.asarray(prediction)
        self.skipped_frames = 0
        self.trace = []


class bugs_tracker:
    def __init__(self) -> None:
        self.tracks = []
        self.dist_thresh = 160
        self.max_frames_to_skip = 30
        self.trackIdcount = 100
        self.max_trace_len = 5
    
    def update(self, detections):
        
        # create tracks when no tracks found
        if len(self.tracks) ==0 :
            for detection in detections:
                obj = Track(detection,self.trackIdcount)
                self.trackIdcount +=1
                self.tracks.append(obj)
        
        # Hungrian matching
        N = len(self.tracks)
        M = len(detections)
        
        cost = np.zeros((N,M), np.float32)
        for i in range(N):
            for j in range(M):
                diff = self.tracks[i].prediction - detections[j]
                distance = np.sqrt(diff[0][0]*diff[0][0] + diff[1][0]*diff[1][0])
                cost[i][j] = distance
        
        assignment = np.ones(N,np.int32)*-1
        
        row_idx, col_idx = sciOpt.linear_sum_assignment(cost)
        for i in range(len(row_idx)):
            assignment[row_idx[i]] = col_idx[i]
        
        #
        for i in range(len(assignment)):
            if assignment[i] != -1:
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
            
            else:
                self.tracks[i].skipped_frames +=1
        
        # remove track that is not detected for long time
        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:
            for id in del_tracks:
                if id < len(self.tracks):
                    self.tracks.pop(id)
                    assignment = np.delete(assignment,id)
        # check unsigned tracks
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
        
        # add unsigned tracks
        if len(un_assigned_detects) != 0 :
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdcount)
                self.trackIdcount+=1
                self.tracks.append(track)
                
        # update tracks using kalman filter
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            
            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.update(detections[assignment[i]],1)
            else:
                self.tracks[i].prediction = self.tracks[i].KF.update(np.array([[0],[0]]), 0)
                
            if len(self.tracks[i].trace) > self.max_trace_len:
                for j in range(len(self.tracks[i].trace) - self.max_trace_len):
                    self.tracks[i].trace.pop(j)
            
            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction
                    
        
        
                    
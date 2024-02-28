import numpy as np



class KalmanFilter():
    def __init__(self) -> None:
        self.lastResult = np.array([[0], [255]])
        self.dt = 0.005 # delta time
        
        self.x = np.zeros((2,1))                        # current state
        self.F = np.array([[1.0,self.dt],[0.0,1.0]])    # prediction transition matrix
        self.P = np.diag((3.0,3.0))                     # current state covariance matrix
        self.Q = np.eye(self.x.shape[0])                # prediction noise covariance matrix
        self.z = np.array([[0], [255]])                 # observation state
        self.H = np.array([[1,0], [0,1]])               # observation matrix
        self.R = np.eye(self.z.shape[0])                # observation noise covariance matrix
        
        
    def update(self, observ, flag):
        
        if not flag:
            self.z = self.lastResult
        else:
            self.z = observ
        
        # cal kalman coeff
        c_inv = np.linalg.inv(np.dot(self.H,np.dot(self.P,self.H.T)) + self.R)
        K = np.dot(self.P, np.dot(self.H.T,c_inv))
        
        # update state
        self.x = np.round(self.x + np.dot(K, (self.z - np.dot(self.H.T, self.x))))
        
        # update covariance matrix
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        
        self.lastResult = self.x
        return self.x
    
    def predict(self):
        
        # state transition (predict state)
        self.x = np.round(np.dot(self.F, self.x))
        
        # covariance transition (predict covariance)
        self.P = np.dot(self.F, np.dot(self.P, self.F.transpose())) + self.Q
        
        
        self.lastResult = self.x
        
        return self.x
    
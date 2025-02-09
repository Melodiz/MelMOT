import numpy as np

class KalmanFilter:
    def __init__(self, dt=1.0, state_dim=4, meas_dim=2):
        self.dt = dt
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Define the state transition matrix
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Define the measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Define the process noise covariance
        self.Q = np.eye(self.state_dim) * 0.1

        # Define the measurement noise covariance
        self.R = np.eye(self.meas_dim) * 1.0

        # Initialize state and covariance
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 1000

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[:2].flatten()
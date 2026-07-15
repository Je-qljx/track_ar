import numpy as np


class LaneKalmanFilter:
    def __init__(self):
        self.state_dim = 3
        self.meas_dim = 1
        self.dt = 1.0 / 60.0
        self.initialized = False
        self.x = np.zeros((3, 1), dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 100.0
        self._build_matrices()

    def set_dt(self, dt: float):
        dt = max(0.001, min(dt, 1.0))
        self.dt = dt
        if self.initialized:
            self._build_matrices()
            if self.P.shape != (3, 3):
                self.P = np.eye(3, dtype=np.float64) * 100.0

    def initialize(self, x_meas: float | np.ndarray, y_meas: float = 0.0):
        if isinstance(x_meas, np.ndarray):
            self.x = np.zeros((3, 1), dtype=np.float64)
            self.x[0, 0] = float(x_meas.flat[0])
        else:
            self.x = np.array([[x_meas], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 100.0
        self.initialized = True

    def _build_matrices(self):
        dt = self.dt
        dt2 = dt * dt
        self.F = np.array([
            [1.0, dt, 0.5 * dt2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        # Process noise: position ~0.5m, velocity ~2m/s, accel ~5m/s^2
        self.Q = np.diag([0.5, 2.0, 5.0]).astype(np.float64)
        # Measurement noise: base 2m, scales with confidence
        self.R_base = np.array([[2.0]], dtype=np.float64)

    def get_adaptive_R(self, confidence: float = 1.0) -> np.ndarray:
        scale = 1.0 + (1.0 - min(confidence, 1.0)) * 10.0
        return self.R_base * scale

    def predict(self):
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x_meas: float | np.ndarray, confidence: float = 1.0):
        if not self.initialized:
            self.initialize(x_meas)
            return
        if isinstance(x_meas, np.ndarray):
            z = np.array([[float(x_meas.flat[0])]], dtype=np.float64)
        else:
            z = np.array([[x_meas]], dtype=np.float64)
        R = self.get_adaptive_R(confidence)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Innovation gating: reject updates with >3-sigma innovation
        innovation = float(y[0, 0])
        sigma = float(np.sqrt(S[0, 0]))
        if abs(innovation) > 3.0 * sigma:
            return
        self.x = self.x + K @ y
        self.P = (np.eye(3, dtype=np.float64) - K @ self.H) @ self.P
        self.x[1, 0] = np.clip(self.x[1, 0], -15.0, 15.0)
        self.x[0, 0] = max(0.0, self.x[0, 0])

    def get_state(self):
        return (float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]))

    def get_position(self):
        return (float(self.x[0, 0]), 0.0)

    def get_velocity(self) -> float:
        return float(self.x[1, 0])

    def get_acceleration(self) -> float:
        return float(self.x[2, 0])

    def get_predicted_position(self, steps: int = 1) -> float:
        pos = float(self.x[0, 0])
        vel = float(self.x[1, 0])
        acc = float(self.x[2, 0])
        dt = self.dt
        for _ in range(steps):
            pos += vel * dt + 0.5 * acc * dt * dt
            vel += acc * dt
        return pos

import numpy as np
import cv2
from .coords import WorldCoord, ImageCoord, TrackGeometry


class Calibrator:
    def __init__(self, camera_matrix: np.ndarray | None = None, image_size: tuple[int, int] = (1920, 1080)):
        self.image_size = image_size
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            fx = image_size[0] * 1.2
            fy = image_size[0] * 1.2
            cx = image_size[0] / 2
            cy = image_size[1] / 2
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.rvec: np.ndarray | None = None
        self.tvec: np.ndarray | None = None

    def solve_pnp(self, world_pts: list[WorldCoord], image_pts: list[ImageCoord]) -> tuple[np.ndarray, np.ndarray]:
        w_pts = np.array([w.as_array for w in world_pts], dtype=np.float64)
        i_pts = np.array([[i.u, i.v] for i in image_pts], dtype=np.float64)
        # Use RANSAC for robustness; fall back to standard solvePnP on failure
        ret, rvec, tvec, _ = cv2.solvePnPRansac(
            w_pts, i_pts, self.camera_matrix, self.dist_coeffs,
            iterationsCount=2000, reprojectionError=8.0, confidence=0.99,
        )
        if not ret:
            ret, rvec, tvec = cv2.solvePnP(w_pts, i_pts, self.camera_matrix, self.dist_coeffs)
            if not ret:
                raise RuntimeError("solvePnP failed. Check reference points.")
        self.rvec = rvec
        self.tvec = tvec
        return rvec, tvec

    def get_projection_error(self, world_pts: list[WorldCoord], image_pts: list[ImageCoord]) -> float:
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Calibrate first.")
        w_pts = np.array([w.as_array for w in world_pts], dtype=np.float64)
        i_pts = np.array([[i.u, i.v] for i in image_pts], dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(w_pts, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        errors = np.sqrt(np.sum((i_pts - proj_pts[:, 0]) ** 2, axis=1))
        return float(np.mean(errors))

    def get_per_point_errors(self, world_pts: list[WorldCoord], image_pts: list[ImageCoord]) -> list[float]:
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Calibrate first.")
        w_pts = np.array([w.as_array for w in world_pts], dtype=np.float64)
        i_pts = np.array([[i.u, i.v] for i in image_pts], dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(w_pts, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        errors = np.sqrt(np.sum((i_pts - proj_pts[:, 0]) ** 2, axis=1))
        return [float(e) for e in errors]

    def print_calibration_debug(self, world_pts: list[WorldCoord], image_pts: list[ImageCoord]):
        w_pts = np.array([w.as_array for w in world_pts], dtype=np.float64)
        i_pts = np.array([[i.u, i.v] for i in image_pts], dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(w_pts, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        names = ["Start x Lane1", "Start x Lane8", "Finish x Lane1", "Finish x Lane8"]
        print("  Per-point reprojection errors:")
        for name, w, ip, pp in zip(names, w_pts, i_pts, proj_pts[:, 0]):
            err = np.hypot(ip[0] - pp[0], ip[1] - pp[1])
            print(f"    {name}: world=({w[0]:.1f},{w[1]:.2f},{w[2]:.1f})  "
                  f"click=({ip[0]:.0f},{ip[1]:.0f})  "
                  f"proj=({pp[0]:.0f},{pp[1]:.0f})  error={err:.1f}px")
        R, _ = cv2.Rodrigues(self.rvec)
        cam_pos = -R.T @ self.tvec
        print(f"  Camera position: x={cam_pos[0,0]:.1f}m, y={cam_pos[1,0]:.1f}m, z={cam_pos[2,0]:.1f}m")

    def is_calibrated(self) -> bool:
        return self.rvec is not None and self.tvec is not None

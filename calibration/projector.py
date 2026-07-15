import numpy as np
import cv2
from .coords import WorldCoord, ImageCoord


class Projector:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray | None = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1), dtype=np.float64)
        self.rvec: np.ndarray | None = None
        self.tvec: np.ndarray | None = None
        self._calib_world_pts: np.ndarray | None = None  # stored for per-frame tracking

    def set_extrinsics(self, rvec: np.ndarray, tvec: np.ndarray):
        self.rvec = rvec
        self.tvec = tvec
        self._camera_position = None

    def set_calibration_world_pts(self, world_pts: list[WorldCoord]):
        self._calib_world_pts = np.array([w.as_array for w in world_pts], dtype=np.float64)

    def track_homography(self, H: np.ndarray | None, min_displacement_px: float = 0.0) -> bool:
        """Update camera extrinsics from tracked homography H (calib frame → current frame).
           Projects the calibration world points through the current extrinsics,
           warps through H, then re-solves PnP to recover the updated 6-DOF pose.
           Returns True if pose was updated."""
        if H is None or self.rvec is None or self.tvec is None or self._calib_world_pts is None:
            return False
        # 1. Project calibration world points through current extrinsics
        calib_2d, _ = cv2.projectPoints(
            self._calib_world_pts, self.rvec, self.tvec,
            self.camera_matrix, self.dist_coeffs)
        # 2. Warp through homography to current frame
        current_2d = cv2.perspectiveTransform(calib_2d, H)
        # 3. Check actual pixel displacement — skip near-identity (prevents drift)
        displacement = float(np.sqrt(np.mean(np.sum((current_2d - calib_2d) ** 2, axis=2))))
        if displacement < min_displacement_px:
            return True  # no significant motion, keep current pose
        # 4. Solve PnP to recover new pose
        ret, rvec, tvec = cv2.solvePnP(
            self._calib_world_pts, current_2d,
            self.camera_matrix, self.dist_coeffs,
            useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
        if ret:
            self.rvec = rvec
            self.tvec = tvec
            self._camera_position = None
            return True
        return False

    def get_camera_position(self) -> np.ndarray:
        if self._camera_position is None:
            R, _ = cv2.Rodrigues(self.rvec)
            self._camera_position = (-R.T @ self.tvec).ravel()
        return self._camera_position

    def look_at(self, target: WorldCoord):
        cam_pos = self.get_camera_position()
        look_at_pt = np.array([target.x, target.y, 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = look_at_pt - cam_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        self.rvec, _ = cv2.Rodrigues(R.T)
        self.tvec = (-R.T @ cam_pos.reshape(3, 1)).astype(np.float64)
        self._camera_position = None

    def project(self, world: WorldCoord) -> ImageCoord:
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Extrinsics not set. Call set_extrinsics first.")
        pts_3d = world.as_array.reshape(1, 1, 3).astype(np.float64)
        pts_2d, _ = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        u, v = pts_2d[0, 0]
        return ImageCoord(u=float(u), v=float(v))

    def project_batch(self, world_pts: list[WorldCoord]) -> list[ImageCoord]:
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Extrinsics not set.")
        pts_3d = np.array([w.as_array for w in world_pts], dtype=np.float64).reshape(-1, 1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs)
        return [ImageCoord(u=float(p[0, 0]), v=float(p[0, 1])) for p in pts_2d]

    def unproject_to_ground(self, img_pt: ImageCoord) -> WorldCoord:
        if self.rvec is None or self.tvec is None:
            raise RuntimeError("Extrinsics not set.")
        R, _ = cv2.Rodrigues(self.rvec)
        inv_K = np.linalg.inv(self.camera_matrix)
        uv_h = np.array([img_pt.u, img_pt.v, 1.0], dtype=np.float64)
        d_cam = inv_K @ uv_h
        d_world = R.T @ d_cam
        C_w = -R.T @ self.tvec.flatten()
        t = -C_w[2] / d_world[2]
        P = C_w + t * d_world
        return WorldCoord(x=float(P[0]), y=float(P[1]), z=0.0)

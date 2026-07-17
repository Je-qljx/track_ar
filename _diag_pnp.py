import numpy as np
import cv2
from calibration.coords import TrackGeometry

K = np.array([[700, 0, 960], [0, 700, 540], [0, 0, 1]], dtype=np.float64)
R0 = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0 = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)

geom = TrackGeometry(track_type="100m")
wp = geom.calibration_world_points()
wa = np.array([w.as_array for w in wp], dtype=np.float64)
calib_2d, _ = cv2.projectPoints(wa, R0, T0, K, np.zeros((4, 1)))

def rotation_around_y(r0, theta):
    r = r0.copy()
    r[1, 0] += theta
    return r

for theta_deg in [5, 10, 15, 20, 30, 46]:
    theta = np.deg2rad(theta_deg)
    rvec_gt = rotation_around_y(R0, theta)
    R_calib, _ = cv2.Rodrigues(R0)
    R_curr, _ = cv2.Rodrigues(rvec_gt)
    R_rel = R_curr @ R_calib.T
    H_true = K @ R_rel @ np.linalg.inv(K)
    H_true = H_true / H_true[2, 2]
    current_2d = cv2.perspectiveTransform(calib_2d, H_true)
    ret, rvec_pnp, tvec_pnp = cv2.solvePnP(
        wa, current_2d, K, np.zeros((4, 1)),
        useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
    r_err = np.linalg.norm(rvec_pnp - rvec_gt)
    t_err = np.linalg.norm(tvec_pnp - T0)
    print(f"theta={theta_deg:2d}deg ({theta:.3f}rad): r_err={r_err:.6f} t_err={t_err:.4f}")
    print(f"  gt_rvec[1]={rvec_gt[1,0]:.4f} pnp_rvec[1]={rvec_pnp[1,0]:.4f}")

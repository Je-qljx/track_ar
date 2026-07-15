import numpy as np
import cv2
from calibration.track_model import TrackModel

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)

# Check 100m original params
R0_100M = np.array([[-1.9865], [-0.7462], [-0.4312]], dtype=np.float64)
T0_100M = np.array([[-40.827], [-14.574], [36.127]], dtype=np.float64)

print("100m original params:")
for d_m in [0, 10, 50, 100]:
    pts = np.array([[d_m, y, 0.0] for y in [0.61, 1.83, 3.05, 4.27, 5.49, 6.71, 7.93, 9.15]], dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, R0_100M, T0_100M, cam_K, np.zeros((4, 1)))
    n_vis = sum(1 for p in proj if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
    print(f'  d_m={d_m}: {n_vis}/8 visible, positions:')
    for lane in range(1, 9):
        p = proj[lane-1][0]
        print(f'    L{lane}: ({p[0]:.0f},{p[1]:.0f})')

# Better 100m params
print("\nBetter 100m params (camera behind start, centered):")
for cx, cy, alt in [(-5, 4.88, 30), (-5, 4.88, 25), (-3, 4.88, 20)]:
    eye = np.array([cx, cy, alt], dtype=np.float64)
    target = np.array([50.0, 4.88, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_w2c = np.column_stack([x_axis, y_axis, z_axis]).T
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ eye.reshape(3, 1)
    pts = np.array([[0.0, y, 0.0] for y in [0.61, 1.83, 3.05, 4.27, 5.49, 6.71, 7.93, 9.15]], dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
    n_vis = sum(1 for p in proj if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
    if n_vis >= 6:
        rv = rvec.ravel()
        tv = tvec.ravel()
        print(f'  eye=({cx},{cy},{alt}): rvec=[{rv[0]:.4f},{rv[1]:.4f},{rv[2]:.4f}], tvec=[{tv[0]:.1f},{tv[1]:.1f},{tv[2]:.1f}]')
        print(f'    {n_vis}/8 at start:')
        for lane in range(1, 9):
            p = proj[lane-1][0]
            print(f'    L{lane}: ({p[0]:.0f},{p[1]:.0f})')

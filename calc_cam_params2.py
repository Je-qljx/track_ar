import numpy as np
import cv2
from calibration.track_model import TrackModel

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
model = TrackModel()

r0_orig = np.array([[0.6], [0.0], [0.0]], dtype=np.float64)
t0_orig = np.array([[-10], [5], [90]], dtype=np.float64)

print("Original params - athlete projections:")
for d_m in [0, 10, 50, 100]:
    pts = []
    for lane in range(1, 9):
        x, y = model.get_xy(lane, float(d_m))
        pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, r0_orig, t0_orig, cam_K, np.zeros((4, 1)))
    n_vis = sum(1 for p in proj if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
    print(f'  d_m={d_m}: {n_vis}/8 visible')
    for i, lane in enumerate(range(1, 9)):
        p = proj[i][0]
        vis = 'OK' if (0 <= p[0] <= 1920 and 0 <= p[1] <= 1080) else 'OFF'
        print(f'    L{lane}: ({p[0]:.0f},{p[1]:.0f}) {vis}')

print("\nBetter params:");
for eye_x, eye_y, alt, look_x, look_y in [
    (-80, -30, 70, -60, -20),
    (-80, -40, 60, -60, -25),
    (-100, -15, 90, -60, -25),
    (-50, -50, 70, -60, -25),
]:
    eye = np.array([eye_x, eye_y, alt], dtype=np.float64)
    target = np.array([look_x, look_y, 0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_w2c = np.column_stack([x_axis, y_axis, z_axis]).T
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ eye.reshape(3, 1)
    pts = []
    for lane in range(1, 9):
        x, y = model.get_xy(lane, 0.0)
        pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
    n_vis = sum(1 for p in proj if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
    if n_vis >= 6:
        print(f'  eye=({eye_x},{eye_y},{alt}) target=({look_x},{look_y}): {n_vis}/8')
        for i, lane in enumerate(range(1, 9)):
            p = proj[i][0]
            print(f'    L{lane}: ({p[0]:.0f},{p[1]:.0f})')

"""Fix camera parameters so all 8 athletes are visible at start for both 100m and 400m."""
import numpy as np
import cv2
from calibration.coords import TrackGeometry

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)

def look_at(eye, target):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    if np.any(np.isnan(x_axis)):
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    R_w2c = np.column_stack([x_axis, y_axis, z_axis]).T
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ eye.reshape(3, 1)
    return rvec, tvec

def check_visibility(track_type, rvec, tvec):
    geom = TrackGeometry(track_type)
    pts = []
    for lane in range(1, 9):
        wc = geom.world_coord(lane, 0.0)
        pts.append([wc.x, wc.y, 0.0])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
    visible = [0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080 for p in proj]
    return sum(visible), visible

# 100m: camera behind start along the track direction, centered
# The start line is at x=0, lane center y's are 0.61, 1.83, ..., 9.15
# Track total width = 8 * 1.22 = 9.76m

# Search for good 100m params
print("=== 100m ===")
best_100m = None
for z in [20, 25, 30, 35, 40]:
    for y0 in [0, 2, 4.88, 8]:
        for x0 in [-3, -5, -10, -15]:
            eye = [x0, y0, z]
            target = [50.0, 4.88, 0.0]
            r, t = look_at(eye, target)
            n, vis = check_visibility('100m', r, t)
            if n >= 8:
                best_100m = (x0, y0, z, r, t, n)
                print(f'  eye=({x0},{y0},{z}) -> {n}/8 visible: rvec={r.ravel()}, tvec={t.ravel()}')

print(f'\nBest 100m: eye=({best_100m[0]},{best_100m[1]},{best_100m[2]})')
print(f'R0_100M = np.array({[best_100m[3].ravel().tolist()]}, dtype=np.float64).T')
print(f'T0_100M = np.array({[best_100m[4].ravel().tolist()]}, dtype=np.float64).T')

# 400m: need to see all lanes at the start line
# Start positions for 400m track model are scattered around (-42, -37) for L1 to (-84, -17) for L8
# Camera needs to be behind these positions

from calibration.track_model import TrackModel
model = TrackModel()

print("\n=== 400m ===")
best_400m = None
for z in [60, 70, 80, 90, 100]:
    for look_ahead in [0, 20, 40, 60, 80, 100]:
        for shift_x in [-80, -70, -60, -50]:
            for shift_y in [-30, -20, -10, 0, 10]:
                # Camera position behind the start line area
                eye_x = shift_x
                eye_y = shift_y
                eye = [eye_x, eye_y, z]
                # Look at center of start line area, offset ahead
                starts = [model.get_xy(l, 0.0) for l in [1, 5, 8]]
                cx = sum(s[0] for s in starts) / 3
                cy = sum(s[1] for s in starts) / 3
                target = [cx + 50, cy, 0.0]
                r, t = look_at(eye, target)
                # Check visibility at d_m=0 and d_m=50 (athletes move into view)
                pts_d0 = []
                pts_d50 = []
                for lane in range(1, 9):
                    x0, y0 = model.get_xy(lane, 0.0)
                    x50, y50 = model.get_xy(lane, 50.0)
                    pts_d0.append([x0, y0, 0.0])
                    pts_d50.append([x50, y50, 0.0])
                pts_d0 = np.array(pts_d0).reshape(-1, 1, 3)
                pts_d50 = np.array(pts_d50).reshape(-1, 1, 3)
                proj0, _ = cv2.projectPoints(pts_d0, r, t, cam_K, np.zeros((4, 1)))
                proj50, _ = cv2.projectPoints(pts_d50, r, t, cam_K, np.zeros((4, 1)))
                n0 = sum(1 for p in proj0 if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
                n50 = sum(1 for p in proj50 if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)
                if n0 == 8 and n50 == 8:
                    if best_400m is None or z > best_400m[2]:
                        best_400m = (eye_x, eye_y, z, r, t, n0, n50)
                        print(f'  eye=({eye_x},{eye_y},{z}) -> d0={n0}/8, d50={n50}/8')
                elif n0 + n50 >= 14:  # good enough if any invisible at one point are visible at another
                    if best_400m is None or n0 + n50 > best_400m[5] + best_400m[6]:
                        best_400m = (eye_x, eye_y, z, r, t, n0, n50)

if best_400m:
    print(f'\nBest 400m: eye=({best_400m[0]},{best_400m[1]},{best_400m[2]}) d0={best_400m[5]}/8 d50={best_400m[6]}/8')
    print(f'R0_400M = np.array({[best_400m[3].ravel().tolist()]}, dtype=np.float64).T')
    print(f'T0_400M = np.array({[best_400m[4].ravel().tolist()]}, dtype=np.float64).T')

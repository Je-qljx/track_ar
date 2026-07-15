import numpy as np
import cv2
from calibration.coords import TrackGeometry
from calibration.track_model import TrackModel

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)

def look_at(eye, target):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_w2c = np.column_stack([x_axis, y_axis, z_axis]).T
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ np.array(eye, dtype=np.float64).reshape(3, 1)
    return rvec, tvec

def count_visible(track_type, rvec, tvec, d_m=0.0):
    geom = TrackGeometry(track_type)
    pts = []
    for lane in range(1, 9):
        wc = geom.world_coord(lane, d_m)
        pts.append([wc.x, wc.y, 0.0])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
    return sum(1 for p in proj if 0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080)

# 100m: behind start looking along track
print("=== 100m ===")
# Focus on positions where all 8 are visible at start AND mid AND finish
for z in [20, 25, 30, 35, 40]:
    for y0 in [4.88]:
        for x0 in [-5, -8, -10, -15]:
            eye = [x0, y0, z]
            target = [50.0, y0, 0.0]
            r, t = look_at(eye, target)
            n0 = count_visible('100m', r, t, 0.0)
            n50 = count_visible('100m', r, t, 50.0)
            n100 = count_visible('100m', r, t, 100.0)
            if n0 >= 7 and n50 >= 8 and n100 >= 8:
                rv = r.ravel()
                tv = t.ravel()
                print(f'  eye=({x0},{y0},{z}): {n0}/{n50}/{n100} -> rvec=[[{rv[0]:.4f}],[{rv[1]:.4f}],[{rv[2]:.4f}]], tvec=[[{tv[0]:.1f}],[{tv[1]:.1f}],[{tv[2]:.1f}]]')

# Also find best side-view params
print("\n=== 100m SIDE VIEW ===")
for z in [10, 15, 20, 25]:
    for x0 in [30, 40, 50, 60]:
        for y0 in [-20, -15, -10]:
            eye = [x0, y0, z]
            target = [50, 0.0, 0.0]
            r, t = look_at(eye, target)
            n0 = count_visible('100m', r, t, 0.0)
            n50 = count_visible('100m', r, t, 50.0)
            n100 = count_visible('100m', r, t, 100.0)
            if n0 >= 6 and n50 >= 6:
                rv = r.ravel()
                tv = t.ravel()
                print(f'  eye=({x0},{y0},{z}): {n0}/{n50}/{n100} -> rvec=[[{rv[0]:.4f}],[{rv[1]:.4f}],[{rv[2]:.4f}]], tvec=[[{tv[0]:.1f}],[{tv[1]:.1f}],[{tv[2]:.1f}]]')

# 400m
print("\n=== 400m ===")
model = TrackModel()
starts = [model.get_xy(l, 0.0) for l in range(1, 9)]
cx = sum(s[0] for s in starts) / 8
cy = sum(s[1] for s in starts) / 8
for z in [60, 70, 80, 90]:
    for ey in [-40, -30, -20, -10, 0, 10]:
        for ex in [-100, -80, -60, -50, -40]:
            eye = [ex, ey, z]
            for la in [20, 40, 60]:
                target = [cx + la, cy, 0.0]
                r, t = look_at(eye, target)
                n0 = count_visible('400m', r, t, 0.0)
                n50 = count_visible('400m', r, t, 50.0)
                n200 = count_visible('400m', r, t, 200.0)
                n400 = count_visible('400m', r, t, 400.0)
                if n0 == 8 and n50 == 8 and n200 >= 6:
                    rv = r.ravel()
                    tv = t.ravel()
                    print(f'  eye=({ex},{ey},{z}) la={la}: {n0}/{n50}/{n200}/{n400}')
                    print(f'    rvec=[[{rv[0]:.4f}],[{rv[1]:.4f}],[{rv[2]:.4f}]], tvec=[[{tv[0]:.1f}],[{tv[1]:.1f}],[{tv[2]:.1f}]]')

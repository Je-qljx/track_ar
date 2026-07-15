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

def check_visible(track_type, rvec, tvec, d_m=0.0):
    geom = TrackGeometry(track_type)
    pts = []
    for lane in range(1, 9):
        wc = geom.world_coord(lane, d_m)
        pts.append([wc.x, wc.y, 0.0])
    pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
    visible = [0 <= p[0,0] <= 1920 and 0 <= p[0,1] <= 1080 for p in proj]
    return sum(visible), [int(p[0,0]) for p in proj], [int(p[0,1]) for p in proj]

# 100m: wide search
print("=== 100m Wide Search ===")
found = []
for z in [15, 20, 25, 30, 35, 40, 45, 50, 60, 80]:
    for y0 in [0, 2, 4.88, 8, 10, 15]:
        for x0 in [-3, -5, -8, -10, -15, -20, -30, -50]:
            for look_x in [30, 50, 80]:
                eye = [x0, y0, z]
                target = [look_x, 4.88, 0.0]
                r, t = look_at(eye, target)
                n0, xs, ys = check_visible('100m', r, t, 0.0)
                if n0 == 8 and all(0 <= x <= 1920 for x in xs) and all(0 <= y <= 1080 for y in ys):
                    found.append((x0, y0, z, r, t, xs, ys))

print(f'Found {len(found)} solutions')
for x0, y0, z, r, t, xs, ys in found[:10]:
    rv = r.ravel()
    tv = t.ravel()
    print(f'  eye=({x0},{y0},{z}): rvec=[{rv[0]:.4f},{rv[1]:.4f},{rv[2]:.4f}] tvec=[{tv[0]:.1f},{tv[1]:.1f},{tv[2]:.1f}]')
    for i, lane in enumerate(range(1, 9)):
        print(f'    L{lane}: ({xs[i]},{ys[i]})')

# 400m wide search
print("\n=== 400m Wide Search ===")
from calibration.track_model import TrackModel
model = TrackModel()

found_400m = []
for z in [50, 60, 70, 80, 90, 100, 110, 120]:
    for ey in [-40, -30, -20, -10, 0, 10, 20, 30]:
        for ex in [-120, -100, -80, -60, -50, -40, -30]:
            eye = [ex, ey, z]
            # look at midpoint of all lane starts
            starts = [model.get_xy(l, 0.0) for l in range(1, 9)]
            cx = sum(s[0] for s in starts) / 8
            cy = sum(s[1] for s in starts) / 8
            for look_ahead in [20, 40, 60, 80, 100]:
                target = [cx + look_ahead, cy, 0.0]
                r, t = look_at(eye, target)
                n0, xs0, ys0 = check_visible('400m', r, t, 0.0)
                if n0 >= 8:
                    found_400m.append((ex, ey, z, r, t, xs0, ys0))

print(f'Found {len(found_400m)} solutions for 400m')
# Show some diverse solutions
shown = set()
for ex, ey, z, r, t, xs, ys in found_400m:
    key = (ex//10*10, ey//10*10, z//10*10)
    if key not in shown:
        shown.add(key)
        rv = r.ravel()
        tv = t.ravel()
        print(f'  eye=({ex},{ey},{z}): rvec=[{rv[0]:.4f},{rv[1]:.4f},{rv[2]:.4f}] tvec=[{tv[0]:.1f},{tv[1]:.1f},{tv[2]:.1f}]')
        for lane in range(1, 9):
            print(f'    L{lane}: ({xs[lane-1]},{ys[lane-1]})')
        if len(shown) >= 10:
            break

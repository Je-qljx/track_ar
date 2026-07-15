"""Find working camera params by manual search, not look_at."""
import numpy as np
import cv2
from calibration.coords import TrackGeometry
from calibration.track_model import TrackModel

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)

def check_vis(track_type, rvec, tvec, dm_list):
    geom = TrackGeometry(track_type)
    results = {}
    for dm in dm_list:
        pts = []
        for lane in range(1, 9):
            wc = geom.world_coord(lane, float(dm))
            pts.append([wc.x, wc.y, 0.0])
        pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, np.zeros((4, 1)))
        results[dm] = [(int(p[0,0]), int(p[0,1])) for p in proj]
    return results

def print_check(track_type, rvec, tvec, label=""):
    dms = [0, 30, 60, 100] if track_type == '100m' else [0, 50, 200, 400]
    res = check_vis(track_type, rvec, tvec, dms)
    all_ok = True
    for dm, pts in res.items():
        visible = [0 <= p[0] <= 1920 and 0 <= p[1] <= 1080 for p in pts]
        n = sum(visible)
        if n < 8:
            all_ok = False
    rv = rvec.ravel()
    tv = tvec.ravel()
    status = "OK" if all_ok else "PARTIAL"
    print(f'{label} [{status}]: rvec=[[{rv[0]:.4f}],[{rv[1]:.4f}],[{rv[2]:.4f}]], tvec=[[{tv[0]:.1f}],[{tv[1]:.1f}],[{tv[2]:.1f}]]')
    for dm, pts in res.items():
        visible = [0 <= p[0] <= 1920 and 0 <= p[1] <= 1080 for p in pts]
        n = sum(visible)
        print(f'  d_m={dm}: {n}/8')
        if n < 8:
            for lane in range(1, 9):
                p = pts[lane-1]
                v = 'OK' if (0 <= p[0] <= 1920 and 0 <= p[1] <= 1080) else 'OFF'
                if v == 'OFF':
                    print(f'    L{lane}: ({p[0]},{p[1]}) {v}')

# 100m params: start at x=0, finish at x=100, y=0 is lane 1 inside edge
# Camera typically on the side, above, looking across the track
print("=== 100m manual params ===")
for angle_down in [0.3, 0.5, 0.7]:
    for angle_side in [0.0, 0.2, 0.4]:
        for pos_x in [-10, -5, 0, 5]:
            for pos_y in [-10, -5, 0, 5]:
                for pos_z in [15, 20, 25, 30]:
                    rvec = np.array([[angle_down], [angle_side], [0.0]], dtype=np.float64)
                    tvec = np.array([[pos_x], [pos_y], [pos_z]], dtype=np.float64)
                    res = check_vis('100m', rvec, tvec, [0, 30, 60, 100])
                    if all(sum(1 for p in pts if 0 <= p[0] <= 1920 and 0 <= p[1] <= 1080) >= 8 for pts in res.values()):
                        print_check('100m', rvec, tvec, f'100m manual ({angle_down},{angle_side})')

# 400m params
print("\n=== 400m manual params ===")
model = TrackModel()
for pitch in [0.3, 0.5, 0.7, 0.9]:
    for yaw in [-0.1, 0.0, 0.1, 0.2]:
        for roll in [-0.1, 0.0, 0.1]:
            for pos_x in [-100, -80, -60, -40, -20]:
                for pos_y in [-20, -10, 0, 10]:
                    for pos_z in [50, 60, 70, 80, 90]:
                        rvec = np.array([[pitch], [yaw], [roll]], dtype=np.float64)
                        tvec = np.array([[pos_x], [pos_y], [pos_z]], dtype=np.float64)
                        res = check_vis('400m', rvec, tvec, [0, 50, 200, 400])
                        if all(sum(1 for p in pts if 0 <= p[0] <= 1920 and 0 <= p[1] <= 1080) >= 8 for pts in res.values()):
                            print_check('400m', rvec, tvec, f'400m manual ({pitch},{yaw},{roll})')

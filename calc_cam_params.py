import numpy as np
import cv2

def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Compute rvec, tvec for a camera looking at target from eye position."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_w2c = np.column_stack([x_axis, y_axis, z_axis]).T  # world→camera
    rvec, _ = cv2.Rodrigues(R_w2c)
    tvec = -R_w2c @ eye.reshape(3, 1)
    return rvec, tvec

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)

# 100m: camera behind start, centered, looking down the track
for name, eye, target in [
    ('100m standard', [-5, 4.88, 30], [50, 4.88, 0]),
    ('100m far', [-10, 4.88, 50], [50, 4.88, 0]),
    ('100m near', [0, 4.88, 15], [50, 4.88, 0]),
    ('400m standard', [-10, 4.88, 50], [100, 4.88, 0]),
    ('400m high', [-10, 4.88, 90], [100, 4.88, 0]),
]:
    rvec, tvec = look_at(eye, target)
    print(f'{name}: rvec={rvec.ravel()}, tvec={tvec.ravel()}')

# Verify: project all 8 lanes at start and finish
rvec_100m, tvec_100m = look_at([-5, 4.88, 30], [50, 4.88, 0])
print(f'\n100m params (for test):')
print(f'rvec={rvec_100m.ravel()}, tvec={tvec_100m.ravel()}')
pts_3d = []
for lane in range(1, 9):
    y = (lane - 0.5) * 1.22
    pts_3d.extend([[0.0, y, 0.0], [100.0, y, 0.0]])
pts_3d = np.array(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
proj, _ = cv2.projectPoints(pts_3d, rvec_100m, tvec_100m, cam_K, np.zeros((4, 1)))
print('Projected start/finish positions:')
for i, lane in enumerate(range(1, 9)):
    start = proj[i*2][0]
    finish = proj[i*2+1][0]
    on_screen_s = 0 <= start[0] <= 1920 and 0 <= start[1] <= 1080
    on_screen_f = 0 <= finish[0] <= 1920 and 0 <= finish[1] <= 1080
    print(f'  L{lane}: start=({start[0]:.0f},{start[1]:.0f}) {"OK" if on_screen_s else "OFF"}, '
          f'finish=({finish[0]:.0f},{finish[1]:.0f}) {"OK" if on_screen_f else "OFF"}')

rvec_400m, tvec_400m = look_at([-10, 4.88, 90], [100, 4.88, 0])
print(f'\n400m params:')
print(f'rvec={rvec_400m.ravel()}, tvec={tvec_400m.ravel()}')
from calibration.track_model import TrackModel
model = TrackModel()
pts_3d = []
for lane in range(1, 9):
    x, y = model.get_xy(lane, 0.0)
    pts_3d.append([x, y, 0.0])
    x, y = model.get_xy(lane, model.race_distance())
    pts_3d.append([x, y, 0.0])
pts_3d = np.array(pts_3d, dtype=np.float64).reshape(-1, 1, 3)
proj, _ = cv2.projectPoints(pts_3d, rvec_400m, tvec_400m, cam_K, np.zeros((4, 1)))
print('Projected start/finish:')
for i, lane in enumerate(range(1, 9)):
    start = proj[i*2][0]
    finish = proj[i*2+1][0]
    on_screen_s = 0 <= start[0] <= 1920 and 0 <= start[1] <= 1080
    on_screen_f = 0 <= finish[0] <= 1920 and 0 <= finish[1] <= 1080
    print(f'  L{lane}: start=({start[0]:.0f},{start[1]:.0f}) {"OK" if on_screen_s else "OFF"}, '
          f'finish=({finish[0]:.0f},{finish[1]:.0f}) {"OK" if on_screen_f else "OFF"}')

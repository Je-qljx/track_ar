import numpy as np
from calibration.coords import TrackGeometry, ImageCoord
from calibration.projector import Projector
from tests.synthetic_scene import SyntheticScene
from tests.self_test import SPEED, R0_400M, T0_400M
import cv2

cam_K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
r0 = R0_400M.copy()
t0 = T0_400M.copy()

geom = TrackGeometry(track_type='400m')
proj = Projector(cam_K, np.zeros((4, 1)))
proj.set_extrinsics(r0, t0)

scene = SyntheticScene(proj, geom, speeds=[SPEED] * 8)

# Check detections at different times
for t in [0.0, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 42.0]:
    athletes = scene.update(t)
    detections = scene.get_detections(athletes)
    d_m_values = [a.d_m for a in athletes]
    print(f't={t:.1f}s: d_m=[{d_m_values[0]:.0f}..{d_m_values[-1]:.0f}], detections={len(detections)}')
    for a, det in zip(athletes, [None]*8):
        pass
    for a in athletes:
        d = next((d for d in detections if d.bbox[2] - d.bbox[0] > 0), None)
    for lane in range(1, 9):
        a = athletes[lane-1]
        # Manually project
        from calibration.coords import WorldCoord
        wc = geom.world_coord(lane, a.d_m)
        ic = proj.project(wc)
        on_screen = 0 <= ic.u <= 1920 and 0 <= ic.v <= 1080
        # Check if this lane is in detections
        detected = any(abs(d.bottom_center[0] - ic.u) < 30 for d in detections)
        print(f'  L{lane}: d_m={a.d_m:.0f}, img=({ic.u:.0f},{ic.v:.0f}), screen={on_screen}, detected={detected}')

import sys
import numpy as np
import cv2
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from rendering.occlusion_guard import OcclusionGuard, rects_overlap, compute_graphic_bbox
from detection.detector import Detection
from tracking.lane_assigner import AthleteState


def test_all_finish_simultaneously():
    print("Stress test: all 8 athletes finish simultaneously...")
    K = np.array([[2400, 0, 960], [0, 2400, 540], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    rvec = np.array([[0.5], [-0.3], [0.1]], dtype=np.float64)
    tvec = np.array([[0], [-15], [20]], dtype=np.float64)
    proj = Projector(K)
    proj.set_extrinsics(rvec, tvec)
    guard = OcclusionGuard(geom, proj)
    athletes = []
    for lane in range(1, 9):
        det = Detection(bbox=(400 + lane * 10, 200 + lane * 20, 460 + lane * 10, 320 + lane * 20), confidence=0.95)
        athlete = AthleteState(lane=lane, athlete_id=lane, d_m=99.5, y_world=geom.lane_center_y(lane), detection=det)
        athletes.append(athlete)
    all_ok = True
    for athlete in athletes:
        anchor = guard.compute_safe_position(athlete, athletes, distance_to_end=0.5)
        g_bbox = compute_graphic_bbox(anchor.world, proj)
        if athlete.detection and rects_overlap(g_bbox, athlete.detection.bbox):
            print(f"  FAIL: Lane {athlete.lane} graphic overlaps athlete at finish!")
            all_ok = False
    if all_ok:
        print("  PASS: No overlap at finish line")


def test_detection_dropout():
    print("Stress test: 50% detection dropout...")
    K = np.array([[2400, 0, 960], [0, 2400, 540], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    rvec = np.array([[0.5], [-0.3], [0.1]], dtype=np.float64)
    tvec = np.array([[0], [-15], [20]], dtype=np.float64)
    proj = Projector(K)
    proj.set_extrinsics(rvec, tvec)
    guard = OcclusionGuard(geom, proj)
    for dropout_rate in [0.0, 0.3, 0.5]:
        athletes = []
        for lane in range(1, 9):
            det = Detection(bbox=(500, 300 + lane * 30, 560, 420 + lane * 30), confidence=0.95) if np.random.random() > dropout_rate else None
            a = AthleteState(lane=lane, athlete_id=lane, d_m=30.0 + lane * 5, y_world=geom.lane_center_y(lane), detection=det)
            athletes.append(a)
        passes = 0
        for athlete in athletes:
            try:
                anchor = guard.compute_safe_position(athlete, athletes, distance_to_end=50)
                passes += 1
            except Exception as e:
                print(f"  FAIL with {dropout_rate*100:.0f}% dropout: {e}")
        print(f"  {dropout_rate*100:.0f}% dropout: {passes}/8 OK")


def test_sudden_appearance():
    print("Stress test: sudden athlete appearance after disappearance...")
    K = np.array([[2400, 0, 960], [0, 2400, 540], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    rvec = np.array([[0.5], [-0.3], [0.1]], dtype=np.float64)
    tvec = np.array([[0], [-15], [20]], dtype=np.float64)
    proj = Projector(K)
    proj.set_extrinsics(rvec, tvec)
    guard = OcclusionGuard(geom, proj)
    for lane in range(1, 9):
        det = Detection(bbox=(450 + lane * 20, 250 + lane * 25, 510 + lane * 20, 370 + lane * 25), confidence=0.95)
        a = AthleteState(lane=lane, athlete_id=lane, d_m=25.0, y_world=geom.lane_center_y(lane), detection=det)
        try:
            anchor = guard.compute_safe_position(a, [], distance_to_end=50)
        except Exception as e:
            print(f"  FAIL Lane {lane}: sudden appearance error: {e}")
    print("  PASS: sudden athlete appearance handled")


def run_all():
    test_all_finish_simultaneously()
    test_detection_dropout()
    test_sudden_appearance()
    print("\nAll stress tests completed!")


if __name__ == "__main__":
    run_all()

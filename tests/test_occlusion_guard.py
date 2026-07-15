import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.coords import TrackGeometry, WorldCoord, ImageCoord
from calibration.projector import Projector
from calibration.calibrator import Calibrator
from detection.detector import Detection
from tracking.lane_assigner import AthleteState
from rendering.occlusion_guard import OcclusionGuard, rects_overlap, compute_graphic_bbox
import cv2


def setup_test_env():
    K = np.array([[2400, 0, 960], [0, 2400, 540], [0, 0, 1]], dtype=np.float64)
    geom = TrackGeometry()
    rvec = np.array([[0.5], [-0.3], [0.1]], dtype=np.float64)
    tvec = np.array([[0], [-15], [20]], dtype=np.float64)
    proj = Projector(K)
    proj.set_extrinsics(rvec, tvec)
    return geom, proj


def test_rects_overlap_no_collision():
    assert not rects_overlap((0, 0, 10, 10), (30, 30, 50, 50), margin_px=0)
    assert not rects_overlap((0, 0, 10, 10), (20, 0, 30, 10), margin_px=0)


def test_rects_overlap_collision():
    assert rects_overlap((0, 0, 10, 10), (5, 5, 15, 15), margin_px=0)
    assert rects_overlap((0, 0, 10, 10), (9, 9, 20, 20), margin_px=0)


def test_rects_overlap_margin():
    assert not rects_overlap((0, 0, 10, 10), (15, 0, 25, 10), margin_px=2)
    assert rects_overlap((0, 0, 10, 10), (12, 0, 22, 10), margin_px=5)


def test_compute_graphic_bbox():
    geom, proj = setup_test_env()
    wc = WorldCoord(20.0, geom.lane_center_y(1), 0.0)
    bbox = compute_graphic_bbox(wc, proj)
    assert len(bbox) == 4
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


def test_occlusion_guard_ahead_placement():
    geom, proj = setup_test_env()
    guard = OcclusionGuard(geom, proj)
    det = Detection(bbox=(500, 400, 560, 520), confidence=0.95)
    athlete = AthleteState(lane=1, athlete_id=1, d_m=20.0, y_world=geom.lane_center_y(1), detection=det)
    all_athletes = [athlete]
    anchor = guard.compute_safe_position(athlete, all_athletes, distance_to_end=80)
    assert anchor.placement_mode == "ahead"
    assert anchor.offset_ahead > 0
    assert anchor.world.x > athlete.d_m


def test_occlusion_guard_behind_at_finish():
    geom, proj = setup_test_env()
    guard = OcclusionGuard(geom, proj)
    det = Detection(bbox=(500, 400, 560, 520), confidence=0.95)
    athlete = AthleteState(lane=1, athlete_id=1, d_m=99.0, y_world=geom.lane_center_y(1), detection=det)
    all_athletes = [athlete]
    anchor = guard.compute_safe_position(athlete, all_athletes, distance_to_end=1.0)
    assert anchor.placement_mode == "behind"
    assert anchor.world.x < athlete.d_m


def test_occlusion_guard_no_overlap():
    geom, proj = setup_test_env()
    guard = OcclusionGuard(geom, proj)
    athletes = []
    for lane in range(1, 5):
        det = Detection(bbox=(400 + lane * 50, 300 + lane * 30, 460 + lane * 50, 420 + lane * 30), confidence=0.95)
        a = AthleteState(lane=lane, athlete_id=lane, d_m=10.0 * lane, y_world=geom.lane_center_y(lane), detection=det)
        athletes.append(a)
    for athlete in athletes:
        anchor = guard.compute_safe_position(athlete, athletes, distance_to_end=50)
        g_bbox = compute_graphic_bbox(anchor.world, proj)
        if athlete.detection:
            overlap = rects_overlap(g_bbox, athlete.detection.bbox)
            assert not overlap, f"Lane {athlete.lane}: graphic overlaps athlete"


def run_all():
    test_rects_overlap_no_collision()
    test_rects_overlap_collision()
    test_rects_overlap_margin()
    test_compute_graphic_bbox()
    test_occlusion_guard_ahead_placement()
    test_occlusion_guard_behind_at_finish()
    test_occlusion_guard_no_overlap()
    print("All occlusion guard tests passed!")


if __name__ == "__main__":
    run_all()

import numpy as np
import cv2


class LaneFeatureTracker:
    """Feature tracker using goodFeaturesToTrack + KLT optical flow.

    Operates at downscaled resolution (max_width=640) for speed.
    Designed for low-texture scenes dominated by line features (track lanes).
    Detects corner features, tracks them frame-to-frame via KLT optical flow,
    and estimates robust homography via USAC_MAGSAC.

    Drift correction design:
    - _first_gray / _first_pts : ORIGINAL calibration frame (never changes).
      Drift correction tracks from original → current → drift-free H.
    - _ref_gray / _ref_pts : rolling reference updated each drift-correction cycle.
      When original-frame tracking fails (camera panned too far), we fall back
      to _ref_gray, which can be refreshed. The cumulative H from original to
      _ref_gray is stored in _H_cumulative.
    - _current_H : H from _ref_gray to current frame.
    - H_total = _current_H @ _H_cumulative.
    """

    def __init__(self, max_width: int = 640, max_features: int = 400,
                 quality_level: float = 0.005, min_distance: float = 3.0,
                 block_size: int = 5):
        self._max_width = max_width
        self._max_features = max_features
        self._quality_level = quality_level
        self._min_distance = min_distance
        self._block_size = block_size
        self._lk_win_size = (21, 21)
        self._lk_max_level = 3
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._homography_reproj_threshold = 3.0

        self._scale = 1.0
        self._first_gray: np.ndarray | None = None   # original calibration (never changes)
        self._first_pts: np.ndarray | None = None     # features on original
        self._prev_gray: np.ndarray | None = None
        self._pts: np.ndarray | None = None           # current tracked features
        self._ref_gray: np.ndarray | None = None      # rolling reference (updated each cycle)
        self._ref_pts: np.ndarray | None = None       # features on _ref_gray

        # H from _ref_gray to current frame
        self._current_H: np.ndarray = np.eye(3, dtype=np.float64)
        # H from original calibration to _ref_gray
        self._H_cumulative: np.ndarray = np.eye(3, dtype=np.float64)

        self._match_info: dict = {}
        self._frame_count: int = 0
        self._min_features_to_keep = 20
        self._redetect_every = 60
        self._last_redetect = 0
        self._consecutive_failures = 0
        self._drift_correct_every = 60
        self._last_drift_correct = 0
        self._drift_threshold = 3.0
        self._first_klt_win_size = (31, 31)
        self._first_klt_max_level = 4

    def _downscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w <= self._max_width:
            self._scale = 1.0
            return gray
        self._scale = self._max_width / w
        new_w = self._max_width
        new_h = int(h * self._scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _to_lr(self, u: float, v: float) -> tuple[float, float]:
        return u * self._scale, v * self._scale

    def _from_lr(self, u: float, v: float) -> tuple[float, float]:
        if self._scale == 0:
            return u, v
        return u / self._scale, v / self._scale

    @property
    def H_calib_current(self) -> np.ndarray:
        return self._current_H @ self._H_cumulative

    @property
    def last_match_info(self) -> dict:
        return self._match_info

    def need_update(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return self._first_gray is not None

    def set_reference(self, gray: np.ndarray):
        lr = self._downscale(gray)
        self._first_gray = lr
        self._first_pts = self._detect_features(lr)
        self._ref_gray = lr.copy()
        self._ref_pts = self._first_pts.copy() if self._first_pts is not None else None
        self._prev_gray = lr.copy()
        self._pts = self._first_pts.copy() if self._first_pts is not None else None
        self._current_H = np.eye(3, dtype=np.float64)
        self._H_cumulative = np.eye(3, dtype=np.float64)
        self._frame_count = 0
        self._last_redetect = 0
        self._last_drift_correct = 0
        self._consecutive_failures = 0

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self._max_features,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            blockSize=self._block_size,
            useHarrisDetector=False)
        return pts

    def _track_klt(self, ref_gray, ref_pts, curr_gray,
                   win_size=(21, 21), max_level=3):
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            ref_gray, curr_gray, ref_pts, None,
            winSize=win_size, maxLevel=max_level,
            criteria=self._lk_criteria)
        if new_pts is not None and status is not None:
            good = status.flatten() == 1
            return new_pts[good].reshape(-1, 1, 2), ref_pts[good].reshape(-1, 1, 2)
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1, 2), dtype=np.float32)

    def _is_valid_homography(self, H: np.ndarray | None) -> bool:
        if H is None:
            return False
        if not np.isfinite(H).all():
            return False
        try:
            det = np.linalg.det(H)
            if abs(det) < 0.01 or abs(det) > 10.0:
                return False
            scale = np.sqrt(abs(det))
            if scale < 0.3 or scale > 3.0:
                return False
            if abs(H[0, 1]) > 2.0 or abs(H[1, 0]) > 2.0:
                return False
            np.linalg.inv(H)
            return True
        except np.linalg.LinAlgError:
            return False

    def _refresh_ref(self, lr_in: np.ndarray):
        """Save current frame as new _ref_gray, accumulate H into _H_cumulative."""
        self._H_cumulative = self._current_H @ self._H_cumulative
        self._current_H = np.eye(3, dtype=np.float64)
        self._ref_gray = lr_in.copy()
        new_pts = self._detect_features(lr_in)
        self._ref_pts = new_pts

    def _compute_drift_correction(self, ref_gray, ref_pts, curr_gray) -> tuple:
        """Compute H_f from ref→current via KLT+USAC. Returns (H_f, inlier_pts, inliers) or (None, None, 0)."""
        ref_curr, ref_src = self._track_klt(
            ref_gray, ref_pts, curr_gray,
            win_size=self._first_klt_win_size, max_level=self._first_klt_max_level)
        if len(ref_curr) >= 8:
            H_f, mask_f = cv2.findHomography(
                ref_src, ref_curr, cv2.USAC_MAGSAC,
                self._homography_reproj_threshold)
            if H_f is not None and mask_f is not None:
                inliers = int(np.sum(mask_f))
                if self._is_valid_homography(H_f):
                    return H_f, ref_curr[mask_f.flatten() == 1], inliers
        return None, None, 0

    def update(self, gray: np.ndarray):
        self._frame_count += 1
        info: dict = {}
        lr_in = self._downscale(gray)

        if self._prev_gray is None or self._pts is None or len(self._pts) < 4:
            self.set_reference(gray)
            info = {'method': 'reset', 'pts': 0}
            self._match_info = info
            return

        # --- KLT tracking (frame-to-frame) ---
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, lr_in, self._pts, None,
            winSize=self._lk_win_size,
            maxLevel=self._lk_max_level,
            criteria=self._lk_criteria)

        if new_pts is not None and status is not None:
            good_new = new_pts[status.flatten() == 1]
            good_old = self._pts[status.flatten() == 1]
        else:
            good_new = np.empty((0, 1, 2), dtype=np.float32)
            good_old = np.empty((0, 1, 2), dtype=np.float32)

        info['klt_tracked'] = len(good_new)
        info['klt_total'] = len(self._pts)

        # --- Homography from KLT tracks ---
        updated = False
        if len(good_new) >= 8:
            H, mask = cv2.findHomography(
                good_old, good_new, cv2.USAC_MAGSAC,
                self._homography_reproj_threshold)
            if H is not None and mask is not None:
                info['klt_inliers'] = int(np.sum(mask))
                if self._is_valid_homography(H):
                    candidate = H @ self._current_H
                    if self._is_valid_homography(candidate):
                        self._current_H = candidate
                        inlier_pts = good_new[mask.flatten() == 1].reshape(-1, 1, 2)
                        self._pts = inlier_pts
                        info['method'] = 'klt_homography'
                        updated = True
                        self._consecutive_failures = 0

        if not updated:
            self._consecutive_failures += 1
            if len(good_new) >= 4:
                self._pts = good_new.reshape(-1, 1, 2)
            else:
                self._pts = None

        self._prev_gray = lr_in

        # --- Redetect if needed ---
        need_redetect = (self._pts is None or len(self._pts) < self._min_features_to_keep
                         or (self._frame_count - self._last_redetect) > self._redetect_every)
        if need_redetect:
            new_pts = self._detect_features(lr_in)
            if new_pts is not None and len(new_pts) > 0:
                if self._pts is not None and len(self._pts) > 0:
                    self._pts = np.vstack([self._pts, new_pts])
                    if len(self._pts) > self._max_features:
                        self._pts = self._pts[:self._max_features]
                else:
                    self._pts = new_pts
            self._last_redetect = self._frame_count

        # --- Drift correction ---
        if ((self._first_pts is not None or self._ref_pts is not None)
                and (self._frame_count - self._last_drift_correct) >= self._drift_correct_every):

            drift_corrected = False

            # Priority 1: track from ORIGINAL calibration frame.
            # H_f maps original→current directly with NO composition drift.
            # Apply it unconditionally — always more accurate than 60-frame composition.
            if self._first_pts is not None and len(self._first_pts) >= 4:
                H_f, inlier_pts, inliers = self._compute_drift_correction(
                    self._first_gray, self._first_pts, lr_in)
                info['first_inliers'] = inliers
                info['first_total'] = len(self._first_pts)
                if H_f is not None:
                    current_total = self._current_H @ self._H_cumulative
                    drift = float(np.sum(np.abs(H_f - current_total)))
                    info['drift_from_first'] = drift
                    # Always trust direct original→current tracking (drift-free)
                    self._current_H = H_f
                    self._H_cumulative = np.eye(3, dtype=np.float64)
                    self._pts = inlier_pts
                    info['drift_orig'] = True
                    drift_corrected = True
                    # Rolling reference = this frame; H_cumulative = I takes care
                    self._ref_gray = lr_in.copy()
                    self._ref_pts = self._detect_features(lr_in)

            # Priority 2: track from rolling reference (original is too far).
            # H_f maps ref→current; H_corrected = H_f @ _H_cumulative maps original→current.
            if not drift_corrected and self._ref_pts is not None and len(self._ref_pts) >= 4:
                H_f, inlier_pts, inliers = self._compute_drift_correction(
                    self._ref_gray, self._ref_pts, lr_in)
                info['ref_inliers'] = inliers
                info['ref_total'] = len(self._ref_pts)
                if H_f is not None:
                    H_corrected = H_f @ self._H_cumulative
                    if self._is_valid_homography(H_corrected):
                        current_total = self._current_H @ self._H_cumulative
                        drift = float(np.sum(np.abs(H_corrected - current_total)))
                        info['drift_from_ref'] = drift
                        if drift > 1.0:  # lower threshold: systematic drift ~0.2 per cycle
                            self._current_H = H_f
                            self._pts = inlier_pts
                            info['drift_ref'] = True
                            drift_corrected = True

            # Priority 3: refresh rolling reference (no correction, just reset window)
            self._ref_gray = lr_in.copy()
            self._ref_pts = self._detect_features(lr_in)
            self._H_cumulative = self._current_H @ self._H_cumulative
            self._current_H = np.eye(3, dtype=np.float64)
            info['ref_cycle'] = True
            info['drift_corrected'] = drift_corrected

            self._last_drift_correct = self._frame_count

        if self._consecutive_failures > 30:
            self._current_H = np.eye(3, dtype=np.float64)
            self._H_cumulative = np.eye(3, dtype=np.float64)
            info['H_reset'] = True
            self._consecutive_failures = 0
            if self._pts is None or len(self._pts) < 4:
                self._pts = self._detect_features(lr_in)

        info['total_pts'] = len(self._pts) if self._pts is not None else 0
        self._match_info = info

    def _safe_H_inv(self) -> np.ndarray:
        try:
            H_total = self._current_H @ self._H_cumulative
            return np.linalg.inv(H_total)
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)

    def current_to_calib(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        H_inv = self._safe_H_inv()
        tr = cv2.perspectiveTransform(pt, H_inv)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))

    def calib_to_current(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        H_total = self._current_H @ self._H_cumulative
        tr = cv2.perspectiveTransform(pt, H_total)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))

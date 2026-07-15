import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class FrameTracker:
    def __init__(self, max_width: int = 640):
        self.orb = cv2.ORB.create(
            nfeatures=1200,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
        )
        self.max_width = max_width
        self._first_frame = True
        self._first_gray: np.ndarray | None = None
        self._first_kp: list[cv2.KeyPoint] | None = None
        self._first_des: np.ndarray | None = None
        self._ref_gray: np.ndarray | None = None
        self._ref_kp: list[cv2.KeyPoint] | None = None
        self._ref_des: np.ndarray | None = None
        self.H_calib_current: np.ndarray = np.eye(3, dtype=np.float64)
        self._scale = 1.0
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_feature: tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray] | None = None

    def _downscale(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w <= self.max_width:
            self._scale = 1.0
            return gray
        self._scale = self.max_width / w
        new_w = self.max_width
        new_h = int(h * self._scale)
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _to_lr(self, u: float, v: float) -> tuple[float, float]:
        return u * self._scale, v * self._scale

    def _from_lr(self, u: float, v: float) -> tuple[float, float]:
        if self._scale == 0:
            return u, v
        return u / self._scale, v / self._scale

    def _compute_features(self, gray: np.ndarray) -> tuple[np.ndarray, list[cv2.KeyPoint], np.ndarray]:
        lr = self._downscale(gray)
        kp, des = self.orb.detectAndCompute(lr, None)
        return lr, kp, des

    def set_reference(self, gray: np.ndarray):
        lr, kp, des = self._compute_features(gray)
        self._first_gray = lr
        self._first_kp = kp
        self._first_des = des
        self._ref_gray = lr
        self._ref_kp = kp[:] if kp else []
        self._ref_des = des.copy() if des is not None else None
        self.H_calib_current = np.eye(3, dtype=np.float64)

    def is_ready(self) -> bool:
        return self._first_gray is not None

    def need_update(self) -> bool:
        if self._first_frame:
            self._first_frame = False
            return True
        return True

    def _filter_matches(self, matches, src_kp, dst_kp, ratio_thresh: float = 0.75) -> list[cv2.DMatch]:
        good = []
        for m in matches:
            if hasattr(m, 'distance') and m.distance < ratio_thresh * 100:
                good.append(m)
        return good

    def _compute_homography(self, src_pts, dst_pts) -> tuple[np.ndarray | None, np.ndarray | None]:
        if len(src_pts) < 6:
            return None, None
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0)
        if H is not None and mask is not None and int(np.sum(mask)) >= 4:
            return H, mask
        return None, None

    def _check_homography_sanity(self, H: np.ndarray) -> bool:
        det = np.linalg.det(H)
        if abs(det) < 0.1 or abs(det) > 10.0:
            return False
        scale = np.sqrt(abs(det))
        if scale < 0.5 or scale > 3.0:
            return False
        # Check that H doesn't imply excessive skew
        if abs(H[0, 1]) > 2.0 or abs(H[1, 0]) > 2.0:
            return False
        return True

    def update(self, gray: np.ndarray):
        if self._first_gray is None:
            self.set_reference(gray)
            return

        lr_in = self._downscale(gray)
        kp_in, des_in = self.orb.detectAndCompute(lr_in, None)
        if des_in is None or len(kp_in) < 12:
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        updated = False
        info: dict[str, int | str] = {}

        # Direct first-frame match
        if self._first_des is not None:
            m0 = bf.match(self._first_des, des_in)
            info['first_matches'] = len(m0)
            if len(m0) >= 6:
                src = np.float32([self._first_kp[m.queryIdx].pt for m in m0]).reshape(-1, 1, 2)
                dst = np.float32([kp_in[m.trainIdx].pt for m in m0]).reshape(-1, 1, 2)
                H0, mask0 = self._compute_homography(src, dst)
                info['first_inliers'] = int(np.sum(mask0)) if mask0 is not None else 0
                if H0 is not None and self._check_homography_sanity(H0):
                    self.H_calib_current = H0
                    info['method'] = 'first_frame'
                    updated = True

        # Pairwise fallback
        if not updated and self._ref_des is not None:
            m_p = bf.match(self._ref_des, des_in)
            info['pairwise_matches'] = len(m_p)
            if len(m_p) >= 6:
                src = np.float32([self._ref_kp[m.queryIdx].pt for m in m_p]).reshape(-1, 1, 2)
                dst = np.float32([kp_in[m.trainIdx].pt for m in m_p]).reshape(-1, 1, 2)
                H_p, mask_p = self._compute_homography(src, dst)
                info['pairwise_inliers'] = int(np.sum(mask_p)) if mask_p is not None else 0
                if H_p is not None and self._check_homography_sanity(H_p):
                    candidate = H_p @ self.H_calib_current
                    if self._check_homography_sanity(candidate):
                        self.H_calib_current = candidate
                        info['method'] = 'pairwise'
                        updated = True

        # Periodic reference reset: re-compute from first frame if accumulated error is large
        if not updated and self._first_des is not None:
            if hasattr(self, '_drift_counter'):
                self._drift_counter += 1
            else:
                self._drift_counter = 1
            info['drift_counter'] = self._drift_counter
            # Try first-frame match with relaxed threshold if we've been drifting
            if self._drift_counter > 30:
                m0 = bf.match(self._first_des, des_in)
                info['reset_matches'] = len(m0)
                if len(m0) >= 4:
                    src = np.float32([self._first_kp[m.queryIdx].pt for m in m0]).reshape(-1, 1, 2)
                    dst = np.float32([kp_in[m.trainIdx].pt for m in m0]).reshape(-1, 1, 2)
                    H0, _ = cv2.findHomography(src, dst, cv2.LMEDS)
                    if H0 is not None and self._check_homography_sanity(H0):
                        self.H_calib_current = H0
                        info['method'] = 'reset_lmeds'
                        updated = True
                self._drift_counter = 0
        else:
            self._drift_counter = 0

        if 'method' not in info:
            info['method'] = 'failed'
        self._match_info = info

        self._ref_gray = lr_in
        self._ref_kp = kp_in
        self._ref_des = des_in

    @property
    def last_match_info(self) -> dict:
        return getattr(self, '_match_info', {})

    def calib_to_current(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        tr = cv2.perspectiveTransform(pt, self.H_calib_current)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))

    def current_to_calib(self, u: float, v: float) -> tuple[float, float]:
        if self._first_gray is None:
            return u, v
        H_inv = np.linalg.inv(self.H_calib_current)
        lu, lv = self._to_lr(u, v)
        pt = np.array([[[lu, lv]]], dtype=np.float64)
        tr = cv2.perspectiveTransform(pt, H_inv)
        return self._from_lr(float(tr[0, 0, 0]), float(tr[0, 0, 1]))

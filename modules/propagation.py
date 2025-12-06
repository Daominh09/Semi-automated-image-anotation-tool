import cv2
import numpy as np
import random
from typing import Tuple, Optional, List, Dict


class AnnotationPropagator:
    DEFAULT_MATCH_RATIO = 0.85
    DEFAULT_MIN_MATCHES = 3
    DEFAULT_NUM_CANDIDATES = 5

    def __init__(
        self,
        prev_sift_params=None,
        next_sift_params=None,
        num_candidates=None,
        segmentation_method="watershed",
        min_segment_overlap=5,
        normalize_for_validation=True,
        validation_target_size=640,
        **kwargs
    ):

        default_sift = dict(
            nfeatures=10000,
            contrastThreshold=0.08,
            edgeThreshold=10,
        )

        self.detector_prev = cv2.SIFT_create(**(prev_sift_params or default_sift))
        self.detector_next = cv2.SIFT_create(**(next_sift_params or default_sift))

        self.match_ratio = self.DEFAULT_MATCH_RATIO
        self.min_matches = self.DEFAULT_MIN_MATCHES
        self.num_candidates = num_candidates or self.DEFAULT_NUM_CANDIDATES

        self.segmentation_method = segmentation_method
        self.min_segment_overlap = min_segment_overlap
        self.normalize_for_validation = normalize_for_validation
        self.validation_target_size = validation_target_size

        self._segment_cache = {}  # Optional caching

    def detect_and_compute(self, image, mask=None, use_prev=False):
        detector = self.detector_prev if use_prev else self.detector_next
        return detector.detectAndCompute(image, mask)

    def extract_object_features(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[List, np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return self.detect_and_compute(image, dilated, use_prev=True)


    def match_features(
        self, desc1: Optional[np.ndarray], desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:

        if desc1 is None or desc2 is None:
            return []

        matches = []

        for i, d1 in enumerate(desc1):
            distances = ((desc2 - d1) ** 2).sum(axis=1)
            best_idx = distances.argmin()
            best = distances[best_idx]
            distances[best_idx] = np.inf
            second = distances.min()

            if second > 0 and best / np.sqrt(second) < self.match_ratio:
                matches.append((np.sqrt(best), i, best_idx))

        unique = {}
        for dist, q, t in matches:
            if t not in unique or dist < unique[t][0]:
                unique[t] = (dist, q, t)

        good = []
        for dist, q, t in unique.values():
            d = cv2.DMatch()
            d.queryIdx = q
            d.trainIdx = t
            d.distance = dist
            good.append(d)

        return sorted(good, key=lambda m: m.distance)

    def estimate_transform_candidates(
        self, kp1: List, kp2: List, matches: List[cv2.DMatch], num_candidates=None
    ) -> List[Tuple[np.ndarray, List[int], int]]:

        k = num_candidates or self.num_candidates

        if len(matches) < self.min_matches:
            return []

        match_list = [(m.distance, m.queryIdx, m.trainIdx) for m in matches]
        candidates = []
        sample_n = 3
        attempts = 100
        threshold = 1.0

        for _ in range(k):
            best_inliers = 0
            best_matrix = None
            best_indices = []

            for _ in range(attempts):
                idxs = (
                    list(range(len(match_list)))
                    if len(match_list) <= sample_n
                    else random.sample(range(len(match_list)), sample_n)
                )
                src = [kp1[match_list[i][1]].pt for i in idxs]
                dst = [kp2[match_list[i][2]].pt for i in idxs]

                M = cv2.getAffineTransform(
                    np.float32(src)[:3], np.float32(dst)[:3]
                )

                inliers = 0
                for dist, qs, qt in match_list:
                    p1 = np.float32(kp1[qs].pt)
                    p2 = np.float32(kp2[qt].pt)
                    p1h = np.array([p1[0], p1[1], 1.0], dtype=np.float32)
                    pred = M @ p1h
                    if np.linalg.norm(pred - p2) < threshold:
                        inliers += 1

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_matrix = M
                    best_indices = idxs

            if best_matrix is not None:
                candidates.append((best_matrix, best_indices, best_inliers))

        return sorted(candidates, key=lambda x: x[2], reverse=True)
    
    def _normalize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.normalize_for_validation:
            return img, 1.0
        h, w = img.shape[:2]
        md = max(h, w)
        if md <= self.validation_target_size:
            return img, 1.0
        scale = self.validation_target_size / md
        nh, nw = int(h * scale), int(w * scale)
        return cv2.resize(img, (nw, nh), cv2.INTER_AREA), scale

    def _denormalize(self, labels: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        return cv2.resize(labels.astype(np.float32), (w, h), cv2.INTER_NEAREST).astype(
            np.int32
        )

    def segment_image(self, img: np.ndarray) -> np.ndarray:
        original = img.shape
        norm, s = self._normalize(img)
        g = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY) if norm.ndim == 3 else norm

        if self.segmentation_method == "watershed":
            labels = self._watershed(norm, g)
        elif self.segmentation_method == "slic":
            labels = self._slic(norm)
        else:
            labels = self._grid(g)

        return self._denormalize(labels, original) if s != 1.0 else labels


    def _watershed(self, img, gray):
        _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        open_ = cv2.morphologyEx(t, cv2.MORPH_OPEN, k, 2)
        sure_bg = cv2.dilate(open_, k, 3)
        dist = cv2.distanceTransform(open_, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unk = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unk == 255] = 0
        m = cv2.watershed(img if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
        m[m == -1] = 0
        return m.astype(np.int32)

    def _slic(self, img):
        try:
            slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=10.0)
            slic.iterate(10)
            return slic.getLabels()
        except Exception:
            return self._watershed(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    def _grid(self, gray):
        h, w = gray.shape
        seg = 20
        out = np.zeros((h, w), np.int32)
        sid = 1
        for i in range(0, h, seg):
            for j in range(0, w, seg):
                out[i : i + seg, j : j + seg] = sid
                sid += 1
        return out

    def evaluate_candidate(self, wmask, segments) -> Dict[str, float]:

        area = np.sum(wmask > 0)
        if area == 0:
            return dict(score=float("inf"), symmetric_diff=float("inf"),
                        false_positives=float("inf"), false_negatives=float("inf"),
                        num_segments=0)

        segs = np.unique(segments[wmask > 0])
        segs = segs[segs > 0]
        if len(segs) == 0:
            return dict(score=float("inf"), symmetric_diff=float("inf"),
                        false_positives=float("inf"), false_negatives=float("inf"),
                        num_segments=0)

        mask2 = np.zeros_like(wmask, np.uint8)
        for s in segs:
            if np.sum((segments == s) & (wmask > 0)) >= self.min_segment_overlap:
                mask2[segments == s] = 255

        fp = np.sum((mask2 > 0) & (wmask == 0))
        fn = np.sum((wmask > 0) & (mask2 == 0))
        diff = fp + fn
        score = diff / area

        return dict(
            score=float(score),
            symmetric_diff=int(diff),
            false_positives=int(fp),
            false_negatives=int(fn),
            num_segments=int(len(segs)),
        )

    def warp_mask(self, mask, M, target_shape):
        h, w = target_shape
        if M.shape == (3, 3):
            warped = cv2.warpPerspective(mask, M, (w, h))
        elif M.shape == (2, 3):
            warped = cv2.warpAffine(mask, M, (w, h))
        else:
            raise ValueError(f"Invalid transformation matrix: {M.shape}")
        return cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)[1]

    def propagate_annotation(self, prev_frame, prev_mask, next_frame):
        metrics = {
            "matches": 0,
            "candidates_generated": 0,
            "best_candidate_idx": -1,
            "best_candidate_score": float("inf"),
            "transform_success": False,
        }

        p = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        n = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) if next_frame.ndim == 3 else next_frame

        kp1, d1 = self.extract_object_features(p, prev_mask)
        kp2, d2 = self.detect_and_compute(n, use_prev=False)

        if d1 is None or d2 is None:
            return np.zeros_like(n), metrics

        matches = self.match_features(d1, d2)
        metrics["matches"] = len(matches)

        candidates = self.estimate_transform_candidates(kp1, kp2, matches)
        metrics["candidates_generated"] = len(candidates)

        if not candidates:
            return np.zeros_like(n), metrics

        segments = self.segment_image(next_frame)

        best_score = float("inf")
        best_mask = None
        best_idx = -1

        for idx, (M, _, _) in enumerate(candidates):
            wmask = self.warp_mask(prev_mask, M, n.shape)
            r = self.evaluate_candidate(wmask, segments)
            if r["score"] < best_score:
                best_score = r["score"]
                best_mask = wmask
                best_idx = idx

        metrics.update(
            best_candidate_idx=best_idx,
            best_candidate_score=best_score,
            transform_success=True,
            segmentation_method=self.segmentation_method,
        )

        return best_mask, metrics
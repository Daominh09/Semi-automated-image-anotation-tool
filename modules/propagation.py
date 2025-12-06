import cv2
import numpy as np
import random
from typing import Tuple, Optional, List, Dict
from collections import deque


class AnnotationPropagator:
    
    # SIFT parameters
    DEFAULT_MATCH_RATIO = 0.85
    DEFAULT_RANSAC_THRESHOLD = 5.0
    DEFAULT_MIN_MATCHES = 3
    DEFAULT_NUM_CANDIDATES = 5
    
    def __init__(self,
                prev_sift_params=None,
                next_sift_params=None,
                num_candidates=None,
                segmentation_method='watershed',
                min_segment_overlap=5,
                normalize_for_validation=True,
                validation_target_size=640,
                **kwargs):

        # Default SIFT params if none provided
        default_prev = dict(nfeatures=10000, contrastThreshold=0.08,
                            edgeThreshold=10)
        default_next = dict(nfeatures=10000, contrastThreshold=0.08,
                            edgeThreshold=10)

        prev_sift_params = prev_sift_params or default_prev
        next_sift_params = next_sift_params or default_next

        # Create TWO separate SIFT detectors
        self.detector_prev = cv2.SIFT_create(**prev_sift_params)
        self.detector_next = cv2.SIFT_create(**next_sift_params)

        self.match_ratio = self.DEFAULT_MATCH_RATIO
        self.min_matches = self.DEFAULT_MIN_MATCHES
        self.num_candidates = num_candidates or self.DEFAULT_NUM_CANDIDATES
        
        # Validation parameters
        self.segmentation_method = segmentation_method
        self.min_segment_overlap = min_segment_overlap
        self.normalize_for_validation = normalize_for_validation
        self.validation_target_size = validation_target_size
        
        # Cache for segmentations
        self._segment_cache = {}

    
    def detect_and_compute(self, image, mask=None, use_prev=False):
        detector = self.detector_prev if use_prev else self.detector_next
        keypoints, descriptors = detector.detectAndCompute(image, mask)
        return keypoints, descriptors

    
    def extract_object_features(self, image: np.ndarray, 
                               mask: np.ndarray) -> Tuple[List, np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        keypoints, descriptors = self.detect_and_compute(image, dilated_mask, use_prev=True)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, 
                      desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []
        
        match_pairs = []
        
        # Loop through each descriptor in desc1 (reference/query image)
        for ref_idx in range(len(desc1)):
            # Find the closest and second closest distances to descriptors in desc2
            best_distance = float('inf')
            best_test_idx = -1
            second_best_distance = float('inf')
            
            for test_idx in range(len(desc2)):
                # Calculate Euclidean distance between descriptors
                distance = np.sqrt(np.sum((desc2[test_idx] - desc1[ref_idx]) ** 2))
                
                # Update best and second best distances
                if distance < best_distance:
                    second_best_distance = best_distance
                    best_distance = distance
                    best_test_idx = test_idx
                elif distance < second_best_distance:
                    second_best_distance = distance
            
            # Apply Lowe's ratio test
            if best_test_idx != -1 and second_best_distance > 0:
                if best_distance / second_best_distance < self.match_ratio:
                    match_pairs.append((best_distance, ref_idx, best_test_idx))
        
        # Ensure one-to-one matching
        unique_match_pairs = {}
        for distance, ref_idx, test_idx in match_pairs:
            if test_idx not in unique_match_pairs:
                unique_match_pairs[test_idx] = (distance, ref_idx, test_idx)
            else:
                if distance < unique_match_pairs[test_idx][0]:
                    unique_match_pairs[test_idx] = (distance, ref_idx, test_idx)
        
        # Convert to cv2.DMatch objects
        good_matches = []
        for distance, ref_idx, test_idx in unique_match_pairs.values():
            dmatch = cv2.DMatch()
            dmatch.queryIdx = ref_idx
            dmatch.trainIdx = test_idx
            dmatch.distance = distance
            good_matches.append(dmatch)
        
        good_matches.sort(key=lambda x: x.distance)
        
        return good_matches
    
    def estimate_transform_candidates(self, kp1: List, kp2: List, 
                                     matches: List[cv2.DMatch],
                                     num_candidates: int = None) -> List[Tuple[np.ndarray, List[int], int]]:
        if num_candidates is None:
            num_candidates = self.num_candidates
            
        if len(matches) < self.min_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            return []
        
        num_attempts_per_candidate = 100
        inlier_threshold = 1.0
        num_sample_points = 3
        
        if len(matches) < num_sample_points:
            print(f"Not enough matches for transformation: {len(matches)} < {num_sample_points}")
            return []
        
        candidates = []
        match_list = [(m.distance, m.queryIdx, m.trainIdx) for m in matches]
        
        # Generate multiple candidates
        for candidate_idx in range(num_candidates):
            best_matrix = None
            best_inlier_count = 0
            best_point_indices = []
            
            for attempt in range(num_attempts_per_candidate):
                if len(match_list) == num_sample_points:
                    sample_indices = list(range(len(match_list)))
                else:
                    sample_indices = random.sample(range(len(match_list)), num_sample_points)
                
                sample_matches = [match_list[i] for i in sample_indices]
                
                src_points = []
                dst_points = []
                for _, src_idx, dst_idx in sample_matches:
                    src_points.append(kp1[src_idx].pt)
                    dst_points.append(kp2[dst_idx].pt)
                
                src_points = np.array(src_points, dtype=np.float32)
                dst_points = np.array(dst_points, dtype=np.float32)
                
                M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
                
                inlier_count = 0
                for idx, (_, src_idx, dst_idx) in enumerate(match_list):
                    src_pt = np.array(kp1[src_idx].pt, dtype=np.float32)
                    dst_pt = np.array(kp2[dst_idx].pt, dtype=np.float32)
                    
                    src_pt_h = np.append(src_pt, 1)
                    pred_pt = M @ src_pt_h
                    
                    distance = np.sqrt(np.sum((dst_pt - pred_pt) ** 2))
                    
                    if distance < inlier_threshold:
                        inlier_count += 1
                
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_matrix = M
                    best_point_indices = sample_indices
            
            if best_matrix is not None:
                candidates.append((best_matrix, best_point_indices, best_inlier_count))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Generated {len(candidates)} transformation candidates")
        if candidates:
            print(f"Best candidate has {candidates[0][2]}/{len(matches)} inliers "
                  f"({100*candidates[0][2]/len(matches):.1f}%)")
        
        return candidates
    
    def _normalize_for_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize image for consistent segmentation"""
        if not self.normalize_for_validation:
            return image, 1.0
        
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.validation_target_size:
            return image, 1.0
        
        scale = self.validation_target_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        normalized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return normalized, scale
    
    def _denormalize_segments(self, labels: np.ndarray, 
                             original_shape: Tuple[int, int]) -> np.ndarray:
        """Scale segment labels back to original size"""
        h, w = original_shape[:2]
        return cv2.resize(labels.astype(np.float32), (w, h), 
                         interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        original_shape = image.shape
        
        # Normalize for consistent segmentation
        norm_image, scale = self._normalize_for_segmentation(image)
        
        if len(norm_image.shape) == 3:
            gray = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = norm_image
        
        # Segment based on method
        if self.segmentation_method == 'watershed':
            labels = self._watershed_segmentation(norm_image, gray)
        elif self.segmentation_method == 'slic':
            labels = self._slic_segmentation(norm_image)
        elif self.segmentation_method == 'grid':
            labels = self._grid_segmentation(gray)
        else:
            raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")
        
        # Denormalize back to original size
        if scale != 1.0:
            labels = self._denormalize_segments(labels, original_shape)
        
        return labels
    
    def _watershed_segmentation(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        if len(image.shape) == 3:
            markers = cv2.watershed(image, markers)
        else:
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)
        
        markers[markers == -1] = 0
        
        return markers.astype(np.int32)
    
    def _slic_segmentation(self, image: np.ndarray) -> np.ndarray:
        try:
            slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=20, ruler=10.0)
            slic.iterate(10)
            labels = slic.getLabels()
            return labels
        except:
            print("SLIC not available, falling back to watershed")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return self._watershed_segmentation(image, gray)
    
    def _grid_segmentation(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        segment_size = 20
        labels = np.zeros((h, w), dtype=np.int32)
        
        segment_id = 1
        for i in range(0, h, segment_size):
            for j in range(0, w, segment_size):
                labels[i:i+segment_size, j:j+segment_size] = segment_id
                segment_id += 1
        
        return labels
    
    def evaluate_candidate(self, warped_mask: np.ndarray,
                          target_segments: np.ndarray) -> Dict[str, float]:
        warped_area = np.sum(warped_mask > 0)
        
        if warped_area == 0:
            return {
                'score': float('inf'),
                'symmetric_diff': float('inf'),
                'false_positives': float('inf'),
                'false_negatives': float('inf'),
                'num_segments': 0
            }
        
        # Find overlapping segments
        overlapping_segments = np.unique(target_segments[warped_mask > 0])
        overlapping_segments = overlapping_segments[overlapping_segments > 0]
        
        if len(overlapping_segments) == 0:
            return {
                'score': float('inf'),
                'symmetric_diff': float('inf'),
                'false_positives': float('inf'),
                'false_negatives': float('inf'),
                'num_segments': 0
            }
        
        # Create mask2 by unioning overlapping segments
        mask2 = np.zeros_like(warped_mask, dtype=np.uint8)
        
        for seg_id in overlapping_segments:
            overlap_count = np.sum((target_segments == seg_id) & (warped_mask > 0))
            
            if overlap_count >= self.min_segment_overlap:
                mask2[target_segments == seg_id] = 255
        
        # Calculate symmetric difference
        false_positives = np.sum((mask2 > 0) & (warped_mask == 0))
        false_negatives = np.sum((warped_mask > 0) & (mask2 == 0))
        symmetric_diff = false_positives + false_negatives
        
        # Scale-invariant score
        score = symmetric_diff / warped_area
        
        return {
            'score': float(score),
            'symmetric_diff': int(symmetric_diff),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'num_segments': int(len(overlapping_segments))
        }
    
    
    def warp_mask(self, mask: np.ndarray, M: np.ndarray, 
                  target_shape: Tuple[int, int]) -> np.ndarray:
        
        height, width = target_shape
        
        if M.shape == (3, 3):
            warped_mask = cv2.warpPerspective(mask, M, (width, height))
        elif M.shape == (2, 3):
            warped_mask = cv2.warpAffine(mask, M, (width, height))
        else:
            raise ValueError(f"Invalid transformation matrix shape: {M.shape}")
        
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
        return warped_mask
    
    def propagate_annotation(self, prev_frame: np.ndarray, 
                           prev_mask: np.ndarray,
                           next_frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        metrics = {
            'matches': 0,
            'candidates_generated': 0,
            'best_candidate_idx': -1,
            'best_candidate_score': float('inf'),
            'transform_success': False
        }
        
        # Convert to grayscale if needed
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) \
                    if len(prev_frame.shape) == 3 else prev_frame
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) \
                    if len(next_frame.shape) == 3 else next_frame
        
        # Extract SIFT features
        kp1, desc1 = self.extract_object_features(prev_gray, prev_mask)
        kp2, desc2 = self.detect_and_compute(next_gray, use_prev=False)
        
        if desc1 is None or desc2 is None:
            print("Feature extraction failed")
            return np.zeros(next_gray.shape, dtype=np.uint8), metrics
        
        # Match features
        matches = self.match_features(desc1, desc2)
        metrics['matches'] = len(matches)
        
        # Generate multiple transformation candidates
        candidates = self.estimate_transform_candidates(kp1, kp2, matches)
        metrics['candidates_generated'] = len(candidates)
        
        if not candidates:
            print("No valid transformation candidates found")
            return np.zeros(next_gray.shape, dtype=np.uint8), metrics
        
        print("Pre-segmenting target frame...")
        segments = self.segment_image(next_frame)
        print(f"Found {len(np.unique(segments))} segments")
        
        best_score = float('inf')
        best_mask = None
        best_idx = -1
        
        print(f"\nEvaluating {len(candidates)} candidates...")
        for idx, (M, point_indices, inlier_count) in enumerate(candidates):
            # Warp the mask using this transformation
            warped_mask = self.warp_mask(prev_mask, M, next_gray.shape)
            
            # Evaluate this candidate using segment overlap
            eval_result = self.evaluate_candidate(warped_mask, segments)
            
            print(f"  Candidate {idx+1}: inliers={inlier_count}, "
                  f"validation_score={eval_result['score']:.4f}, "
                  f"segments={eval_result['num_segments']}")
            
            if eval_result['score'] < best_score:
                best_score = eval_result['score']
                best_mask = warped_mask
                best_idx = idx
        
        print(f"Selected candidate {best_idx+1} with validation score {best_score:.4f}")
        
        metrics['transform_success'] = True
        metrics['best_candidate_idx'] = best_idx
        metrics['best_candidate_score'] = best_score
        metrics['best_point_indices'] = candidates[best_idx][1] if best_idx >= 0 else []
        metrics['segmentation_method'] = self.segmentation_method
        
        return best_mask, metrics
    

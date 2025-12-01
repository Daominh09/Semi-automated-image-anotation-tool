import cv2
import numpy as np
import random
from typing import Tuple, Optional, List


class AnnotationPropagator:
    """Propagates annotations between frames using SIFT feature matching"""
    
    # SIFT parameters
    DEFAULT_MATCH_RATIO = 0.85
    DEFAULT_RANSAC_THRESHOLD = 5.0
    DEFAULT_MIN_MATCHES = 3
    
    def __init__(self,
                prev_sift_params=None,
                next_sift_params=None):

        # Default SIFT params if none provided
        default_prev = dict(nfeatures=10000, contrastThreshold=0.03,
                            edgeThreshold=10)
        default_next = dict(nfeatures=10000, contrastThreshold=0.03,
                            edgeThreshold=10)

        prev_sift_params = prev_sift_params or default_prev
        next_sift_params = next_sift_params or default_next

        # Create TWO separate SIFT detectors
        self.detector_prev = cv2.SIFT_create(**prev_sift_params)
        self.detector_next = cv2.SIFT_create(**next_sift_params)

        # Other parameters
        self.match_ratio = self.DEFAULT_MATCH_RATIO
        self.min_matches = self.DEFAULT_MIN_MATCHES

    
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
            # Find the closest and second closest distances to descriptors in desc2 (train image)
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
            # Only keep match if best distance is significantly better than second best
            if best_test_idx != -1 and second_best_distance > 0:
                if best_distance / second_best_distance < self.match_ratio:
                    match_pairs.append((best_distance, ref_idx, best_test_idx))
        
        # Ensure one-to-one matching: each descriptor in desc2 should match only once
        unique_match_pairs = {}
        for distance, ref_idx, test_idx in match_pairs:
            # Store match pair if it's the first time we find this test keypoint
            if test_idx not in unique_match_pairs:
                unique_match_pairs[test_idx] = (distance, ref_idx, test_idx)
            # If this test keypoint already has a match, keep the one with smaller distance
            else:
                if distance < unique_match_pairs[test_idx][0]:
                    unique_match_pairs[test_idx] = (distance, ref_idx, test_idx)
        
        # Convert to cv2.DMatch objects for compatibility with OpenCV functions
        good_matches = []
        for distance, ref_idx, test_idx in unique_match_pairs.values():
            dmatch = cv2.DMatch()
            dmatch.queryIdx = ref_idx
            dmatch.trainIdx = test_idx
            dmatch.distance = distance
            good_matches.append(dmatch)
        
        # Sort by distance (best matches first)
        good_matches.sort(key=lambda x: x.distance)
        
        return good_matches
    
    def estimate_transform(self, kp1: List, kp2: List, 
                          matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], List[int]]:
        if len(matches) < self.min_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            return None, []
        
        # Number of RANSAC iterations
        num_attempts = 100
        
        # Distance threshold for considering a match as inlier (in pixels)
        inlier_threshold = 1.0
        
        # Number of points needed for transformation
        num_sample_points = 3
        
        if len(matches) < num_sample_points:
            print(f"Not enough matches for transformation: {len(matches)} < {num_sample_points}")
            return None, []
        
        best_matrix = None
        best_inlier_count = 0
        best_point_indices = []  
        
        # Convert matches to list of tuples for easier sampling
        match_list = [(m.distance, m.queryIdx, m.trainIdx) for m in matches]
        
        for attempt in range(num_attempts):
            # Randomly sample points for computing transformation
            if len(match_list) == num_sample_points:
                sample_indices = list(range(len(match_list)))
            else:
                sample_indices = random.sample(range(len(match_list)), num_sample_points)
            
            sample_matches = [match_list[i] for i in sample_indices]
            
            # Extract point coordinates for sampled matches
            src_points = []
            dst_points = []
            for _, src_idx, dst_idx in sample_matches:
                src_points.append(kp1[src_idx].pt)
                dst_points.append(kp2[dst_idx].pt)
            
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            # Compute affine transformation (2x3 matrix)
            M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
            
            # Count inliers by testing transformation on all matches
            inlier_count = 0
            
            for idx, (_, src_idx, dst_idx) in enumerate(match_list):
                # Get source and destination points
                src_pt = np.array(kp1[src_idx].pt, dtype=np.float32)
                dst_pt = np.array(kp2[dst_idx].pt, dtype=np.float32)
                
                # For affine, use affine transformation
                src_pt_h = np.append(src_pt, 1)  # Add homogeneous coordinate
                pred_pt = M @ src_pt_h
                
                # Calculate reprojection error
                distance = np.sqrt(np.sum((dst_pt - pred_pt) ** 2))
                
                # Check if this match is an inlier
                if distance < inlier_threshold:
                    inlier_count += 1
            
            # Update best transformation if this one has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_matrix = M
                best_point_indices = sample_indices  # Store the indices used for this matrix
        
        if best_matrix is None:
            print("Transformation estimation failed - no valid matrix found")
            return None, []
        
        print(f"Transformation estimated with {best_inlier_count}/{len(matches)} inliers "
              f"({100*best_inlier_count/len(matches):.1f}%)")
        
        return best_matrix, best_point_indices
    
    def warp_mask(self, mask: np.ndarray, M: np.ndarray, 
                  target_shape: Tuple[int, int]) -> np.ndarray:
        
        height, width = target_shape
        
        # Check if using homography (3x3) or affine (2x3)
        if M.shape == (3, 3):
            # Homography transformation (perspective warp)
            warped_mask = cv2.warpPerspective(mask, M, (width, height))
        elif M.shape == (2, 3):
            # Affine transformation
            warped_mask = cv2.warpAffine(mask, M, (width, height))
        else:
            raise ValueError(f"Invalid transformation matrix shape: {M.shape}. "
                           f"Expected (2,3) for affine or (3,3) for homography")
        
        # Ensure binary mask (threshold at 127 to handle interpolation artifacts)
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
        return warped_mask
    
    def propagate_annotation(self, prev_frame: np.ndarray, 
                           prev_mask: np.ndarray,
                           next_frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        metrics = {
            'matches': 0,
            'inliers': 0,
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
            # Return empty mask with correct shape
            return np.zeros(next_gray.shape, dtype=np.uint8), metrics
        
        # Match features
        matches = self.match_features(desc1, desc2)
        metrics['matches'] = len(matches)
        
        # Estimate transformation
        M, best_point_indices = self.estimate_transform(kp1, kp2, matches)
        
        if M is None:
            print("Transform estimation failed")
            # Return empty mask with correct shape
            return np.zeros(next_gray.shape, dtype=np.uint8), metrics
        
        metrics['transform_success'] = True
        metrics['best_point_indices'] = best_point_indices
        
        # Warp and return mask
        warped_mask = self.warp_mask(prev_mask, M, next_gray.shape)
        return warped_mask, metrics
    


"""
Person D: Feature Matching & Annotation Propagation Module
Handles SIFT/ORB feature matching and annotation transfer between frames
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class AnnotationPropagator:
    """Propagates annotations between frames using feature matching"""
    
    # Recommended default parameters
    DEFAULT_FEATURE_TYPE = 'ORB'  # 'ORB' or 'SIFT'
    DEFAULT_MATCH_RATIO = 0.75    # Lowe's ratio test threshold
    DEFAULT_RANSAC_THRESHOLD = 5.0
    DEFAULT_MIN_MATCHES = 10
    
    def __init__(self, feature_type: str = 'ORB'):
        """
        Initialize propagator with specified feature detector
        
        Args:
            feature_type: 'ORB' or 'SIFT'
        """
        self.feature_type = feature_type.upper()
        self.match_ratio = self.DEFAULT_MATCH_RATIO
        self.ransac_threshold = self.DEFAULT_RANSAC_THRESHOLD
        self.min_matches = self.DEFAULT_MIN_MATCHES
        
        # Initialize feature detector
        if self.feature_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.feature_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def detect_and_compute(self, image: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Input grayscale image
            mask: Optional mask to restrict feature detection
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, mask)
        return keypoints, descriptors
    
    def extract_object_features(self, image: np.ndarray, 
                               mask: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract features only from the object region (inside/near mask)
        
        Args:
            image: Input grayscale image
            mask: Binary mask of object
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Dilate mask to include surrounding area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Detect features in dilated region
        keypoints, descriptors = self.detect_and_compute(image, dilated_mask)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, 
                      desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches after ratio test
        """
        if desc1 is None or desc2 is None:
            return []
        
        # KNN matching with k=2
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_transform(self, kp1: List, kp2: List, 
                          matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """
        Estimate homography transformation between matched keypoints
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of good matches
            
        Returns:
            3x3 homography matrix or None if estimation fails
        """
        if len(matches) < self.min_matches:
            print(f"Not enough matches: {len(matches)} < {self.min_matches}")
            return None
        
        # Extract matched keypoint coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 
                                     self.ransac_threshold)
        
        if H is None:
            print("Homography estimation failed")
            return None
        
        # Count inliers
        inliers = np.sum(mask)
        print(f"Homography estimated with {inliers}/{len(matches)} inliers")
        
        return H
    
    def warp_mask(self, mask: np.ndarray, H: np.ndarray, 
                  target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warp mask using homography transformation
        
        Args:
            mask: Binary mask to warp
            H: 3x3 homography matrix
            target_shape: (height, width) of target image
            
        Returns:
            Warped mask
        """
        height, width = target_shape
        warped_mask = cv2.warpPerspective(mask, H, (width, height))
        
        # Threshold to ensure binary mask
        _, warped_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
        
        return warped_mask
    
    def refine_with_region_growing(self, image: np.ndarray, 
                                   initial_mask: np.ndarray,
                                   region_grower) -> np.ndarray:
        """
        Refine propagated mask using region growing
        
        Args:
            image: Target image
            initial_mask: Warped mask from previous frame
            region_grower: RegionGrowing object from segmentation module
            
        Returns:
            Refined mask
        """
        # Find seeds inside the initial mask (use mask center points)
        contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return initial_mask
        
        # Get seeds from largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return initial_mask
        
        # Center of mass as seed
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Clear previous seeds and add new one
        region_grower.clear_seeds()
        region_grower.add_seed(cx, cy)
        
        # Perform region growing
        refined_mask, _ = region_grower.segment(image, smooth=True)
        
        return refined_mask
    
    def propagate_annotation(self, prev_frame: np.ndarray, 
                           prev_mask: np.ndarray,
                           next_frame: np.ndarray,
                           refine: bool = True,
                           region_grower=None) -> Tuple[np.ndarray, dict]:
        """
        Main function: Propagate annotation from one frame to next
        
        Args:
            prev_frame: Previous frame (BGR or grayscale)
            prev_mask: Binary mask from previous frame
            next_frame: Next frame to propagate annotation to
            refine: Whether to refine with region growing
            region_grower: RegionGrowing object (required if refine=True)
            
        Returns:
            Tuple of (propagated mask, metrics dict)
        """
        metrics = {
            'matches': 0,
            'inliers': 0,
            'transform_success': False,
            'refinement_applied': False
        }
        
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(next_frame.shape) == 3:
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        else:
            next_gray = next_frame
        
        # Extract features from object region in previous frame
        kp1, desc1 = self.extract_object_features(prev_gray, prev_mask)
        
        # Extract features from entire next frame
        kp2, desc2 = self.detect_and_compute(next_gray)
        
        if desc1 is None or desc2 is None:
            print("Feature extraction failed")
            return np.zeros_like(prev_mask), metrics
        
        # Match features
        matches = self.match_features(desc1, desc2)
        metrics['matches'] = len(matches)
        
        # Estimate transformation
        H = self.estimate_transform(kp1, kp2, matches)
        
        if H is None:
            print("Transform estimation failed, returning empty mask")
            return np.zeros_like(prev_mask), metrics
        
        metrics['transform_success'] = True
        
        # Warp mask to next frame
        warped_mask = self.warp_mask(prev_mask, H, next_gray.shape)
        
        # Refine with region growing if requested
        if refine and region_grower is not None:
            warped_mask = self.refine_with_region_growing(next_frame, 
                                                         warped_mask, 
                                                         region_grower)
            metrics['refinement_applied'] = True
        
        return warped_mask, metrics
    
    def visualize_matches(self, img1: np.ndarray, kp1: List,
                         img2: np.ndarray, kp2: List,
                         matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Visualize feature matches between two images
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Visualization image
        """
        # Draw only top N matches for clarity
        top_matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        return match_img


# Example usage and testing
if __name__ == "__main__":
    print("AnnotationPropagator module loaded successfully!")
    print(f"Recommended defaults:")
    print(f"  Feature type: {AnnotationPropagator.DEFAULT_FEATURE_TYPE}")
    print(f"  Match ratio: {AnnotationPropagator.DEFAULT_MATCH_RATIO}")
    print(f"  RANSAC threshold: {AnnotationPropagator.DEFAULT_RANSAC_THRESHOLD}")
    print(f"  Minimum matches: {AnnotationPropagator.DEFAULT_MIN_MATCHES}")
    
    # Example usage:
    # propagator = AnnotationPropagator(feature_type='ORB')
    # next_mask, metrics = propagator.propagate_annotation(
    #     prev_frame, prev_mask, next_frame, 
    #     refine=True, region_grower=rg
    # )
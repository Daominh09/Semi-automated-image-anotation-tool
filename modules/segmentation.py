"""
Person B: Segmentation & Region Growing Module
Handles region growing segmentation and multi-click correction
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class RegionGrowing:
    """Implements region growing segmentation with multi-click support"""
    
    # Recommended default parameters
    DEFAULT_COLOR_THRESHOLD = 20  # For color images
    DEFAULT_GRAY_THRESHOLD = 15   # For grayscale images
    DEFAULT_MORPH_KERNEL = 5
    
    def __init__(self):
        """Initialize region growing segmentor"""
        self.color_threshold = self.DEFAULT_COLOR_THRESHOLD
        self.gray_threshold = self.DEFAULT_GRAY_THRESHOLD
        self.morph_kernel = self.DEFAULT_MORPH_KERNEL
        self.seeds = []  # Store multiple seed points
        self.current_mask = None
    
    def add_seed(self, x: int, y: int):
        """
        Add a seed point for segmentation
        
        Args:
            x, y: Coordinates of seed point
        """
        self.seeds.append((x, y))
    
    def clear_seeds(self):
        """Clear all seed points"""
        self.seeds = []
        self.current_mask = None
    
    def undo_last_seed(self):
        """Remove the last added seed"""
        if self.seeds:
            self.seeds.pop()
    
    def _is_similar(self, pixel1: np.ndarray, pixel2: np.ndarray, 
                    threshold: float, is_color: bool) -> bool:
        """
        Check if two pixels are similar based on threshold
        
        Args:
            pixel1, pixel2: Pixel values to compare
            threshold: Similarity threshold
            is_color: Whether pixels are color (3 channels) or grayscale
            
        Returns:
            True if pixels are similar
        """
        if is_color:
            # Euclidean distance in RGB space
            distance = np.sqrt(np.sum((pixel1.astype(float) - pixel2.astype(float)) ** 2))
        else:
            # Absolute difference for grayscale
            distance = abs(float(pixel1) - float(pixel2))
        
        return distance < threshold
    
    def region_grow_single_seed(self, image: np.ndarray, seed: Tuple[int, int],
                               threshold: Optional[float] = None) -> np.ndarray:
        """
        Perform region growing from a single seed point
        
        Args:
            image: Input image (BGR or grayscale)
            seed: (x, y) coordinates of seed point
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Binary mask of segmented region
        """
        height, width = image.shape[:2]
        is_color = len(image.shape) == 3
        
        # Use appropriate threshold
        if threshold is None:
            threshold = self.color_threshold if is_color else self.gray_threshold
        
        # Initialize mask and visited array
        mask = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        # Check seed validity
        x, y = seed
        if x < 0 or x >= width or y < 0 or y >= height:
            print(f"Warning: Seed ({x}, {y}) is out of bounds")
            return mask
        
        # Get seed pixel value
        seed_value = image[y, x]
        
        # Queue-based region growing (BFS)
        queue = deque([(x, y)])
        visited[y, x] = True
        
        # 8-connectivity neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                    (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Limit region size to prevent runaway growth
        max_region_size = width * height // 2
        region_size = 0
        
        while queue and region_size < max_region_size:
            cx, cy = queue.popleft()
            
            # Add to mask if similar to seed
            current_value = image[cy, cx]
            if self._is_similar(seed_value, current_value, threshold, is_color):
                mask[cy, cx] = 255
                region_size += 1
                
                # Check neighbors
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    
                    # Check bounds and if not visited
                    if (0 <= nx < width and 0 <= ny < height and 
                        not visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((nx, ny))
        
        if region_size >= max_region_size:
            print(f"Warning: Region growing stopped at maximum size ({max_region_size} pixels)")
        
        return mask
    
    def region_grow_multi_seed(self, image: np.ndarray, 
                              seeds: List[Tuple[int, int]],
                              threshold: Optional[float] = None) -> np.ndarray:
        """
        Perform region growing from multiple seed points
        
        Args:
            image: Input image
            seeds: List of (x, y) seed coordinates
            threshold: Similarity threshold
            
        Returns:
            Combined binary mask from all seeds
        """
        if not seeds:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Grow region from each seed and combine
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for seed in seeds:
            seed_mask = self.region_grow_single_seed(image, seed, threshold)
            combined_mask = cv2.bitwise_or(combined_mask, seed_mask)
        
        return combined_mask
    
    def smooth_mask(self, mask: np.ndarray, 
                    kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Smooth mask using morphological operations
        
        Args:
            mask: Binary mask
            kernel_size: Kernel size for morphological operations
            
        Returns:
            Smoothed mask
        """
        if kernel_size is None:
            kernel_size = self.morph_kernel
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def extract_contours(self, mask: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract contours from binary mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Tuple of (contours list, hierarchy)
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    
    def get_largest_contour(self, contours: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Get the largest contour by area
        
        Args:
            contours: List of contours
            
        Returns:
            Largest contour or None if list is empty
        """
        if not contours:
            return None
        
        return max(contours, key=cv2.contourArea)
    
    def segment(self, image: np.ndarray, 
                threshold: Optional[float] = None,
                smooth: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Main segmentation function using current seeds
        
        Args:
            image: Input image
            threshold: Similarity threshold for region growing
            smooth: Whether to apply morphological smoothing
            
        Returns:
            Tuple of (binary mask, list of contours)
        """
        if not self.seeds:
            # No seeds, return empty mask
            empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            return empty_mask, []
        
        # Perform region growing with all seeds
        mask = self.region_grow_multi_seed(image, self.seeds, threshold)
        
        # Smooth the mask if requested
        if smooth:
            mask = self.smooth_mask(mask)
        
        # Extract contours
        contours, _ = self.extract_contours(mask)
        
        # Store current mask
        self.current_mask = mask
        
        return mask, contours
    
    def draw_contours_on_image(self, image: np.ndarray, 
                               contours: List[np.ndarray],
                               color: Tuple[int, int, int] = (0, 255, 0),
                               thickness: int = 2) -> np.ndarray:
        """
        Draw contours on image
        
        Args:
            image: Input image
            contours: List of contours to draw
            color: BGR color for contours
            thickness: Line thickness
            
        Returns:
            Image with contours drawn
        """
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def create_overlay(self, image: np.ndarray, mask: np.ndarray,
                      alpha: float = 0.3,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Create semi-transparent overlay of mask on image
        
        Args:
            image: Input image
            mask: Binary mask
            alpha: Transparency (0-1)
            color: BGR color for overlay
            
        Returns:
            Image with overlay
        """
        overlay = image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("RegionGrowing module loaded successfully!")
    print(f"Recommended defaults:")
    print(f"  Color threshold: {RegionGrowing.DEFAULT_COLOR_THRESHOLD}")
    print(f"  Grayscale threshold: {RegionGrowing.DEFAULT_GRAY_THRESHOLD}")
    print(f"  Morphological kernel: {RegionGrowing.DEFAULT_MORPH_KERNEL}")
    
    # Example usage:
    # rg = RegionGrowing()
    # rg.add_seed(100, 100)  # Add first seed
    # rg.add_seed(150, 150)  # Add correction seed
    # mask, contours = rg.segment(image)
    # result = rg.draw_contours_on_image(image, contours)
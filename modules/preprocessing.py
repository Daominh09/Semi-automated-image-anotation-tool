"""
Person A: Edge Detection & Preprocessing Module
Responsible for image loading, smoothing, and edge detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EdgeDetector:
    """Handles all edge detection and preprocessing operations"""
    
    # Recommended default parameters
    DEFAULT_GAUSSIAN_SIGMA = 1.5
    DEFAULT_CANNY_LOW = 30
    DEFAULT_CANNY_HIGH = 100
    DEFAULT_SOBEL_KERNEL = 3
    DEFAULT_ITERATIONS = 2
    
    def __init__(self):
        """Initialize the edge detector with default parameters"""
        self.gaussian_sigma = self.DEFAULT_GAUSSIAN_SIGMA
        self.canny_low = self.DEFAULT_CANNY_LOW
        self.canny_high = self.DEFAULT_CANNY_HIGH
        self.sobel_kernel = self.DEFAULT_SOBEL_KERNEL
        self.iterations = self.DEFAULT_ITERATIONS
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
        return image
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_gaussian_smoothing(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        if sigma is None:
            sigma = self.gaussian_sigma
        
        # Calculate kernel size from sigma (must be odd)
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def canny_edge_detection(self, image: np.ndarray, 
                            low_threshold: Optional[int] = None,
                            high_threshold: Optional[int] = None) -> np.ndarray:

        if low_threshold is None:
            low_threshold = self.canny_low
        if high_threshold is None:
            high_threshold = self.canny_high
        
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def sobel_edge_detection(self, image: np.ndarray, 
                            kernel_size: Optional[int] = None) -> np.ndarray:
        if kernel_size is None:
            kernel_size = self.sobel_kernel
        
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        return magnitude
    
    def iterative_canny(self, image: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
        if iterations is None:
            iterations = self.iterations
        
        combined_edges = np.zeros_like(image, dtype=np.uint8)
        
        for i in range(iterations):
            # Increase smoothing with each iteration
            sigma = self.gaussian_sigma * (i + 1)
            smoothed = self.apply_gaussian_smoothing(image, sigma)
            edges = self.canny_edge_detection(smoothed)
            
            # Combine edges (OR operation)
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges
    
    def morphological_closing(self, edge_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
    
    def neighborhood_cleanup(self, edge_map: np.ndarray, threshold: int = 4) -> np.ndarray:
        padded = np.pad(edge_map, 1, mode='constant', constant_values=0)
        cleaned = np.zeros_like(edge_map)
        
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                # Count edge neighbors (8-connectivity)
                neighborhood = padded[i-1:i+2, j-1:j+2]
                neighbor_sum = np.sum(neighborhood > 0) - (padded[i, j] > 0)
                
                # Keep pixel if it has enough edge neighbors
                if neighbor_sum >= threshold:
                    cleaned[i-1, j-1] = 255
        
        return cleaned
    
    def get_clean_edges(self, image: np.ndarray, 
                       method: str = 'canny',
                       cleanup_method: str = 'morphological') -> np.ndarray:
        gray = self.to_grayscale(image)
        
        # Apply smoothing
        smoothed = self.apply_gaussian_smoothing(gray)
        
        # Edge detection
        if method == 'canny':
            edges = self.canny_edge_detection(smoothed)
        elif method == 'sobel':
            sobel_edges = self.sobel_edge_detection(smoothed)
            # Threshold Sobel result
            _, edges = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)
        elif method == 'iterative':
            edges = self.iterative_canny(gray)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Cleanup
        if cleanup_method == 'morphological':
            cleaned = self.morphological_closing(edges)
        elif cleanup_method == 'neighborhood':
            cleaned = self.neighborhood_cleanup(edges)
        else:
            cleaned = edges
        
        return cleaned


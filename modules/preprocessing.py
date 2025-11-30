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
    DEFAULT_CANNY_LOW = 10
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
        """
        Load an image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            BGR image as numpy array, or None if loading fails
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
        return image
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale
        
        Args:
            image: BGR image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_gaussian_smoothing(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply Gaussian smoothing to reduce noise
        
        Args:
            image: Input grayscale image
            sigma: Standard deviation for Gaussian kernel (uses default if None)
            
        Returns:
            Smoothed image
        """
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
        """
        Apply Canny edge detection
        
        Args:
            image: Input grayscale image
            low_threshold: Lower threshold for hysteresis (uses default if None)
            high_threshold: Upper threshold for hysteresis (uses default if None)
            
        Returns:
            Binary edge map
        """
        if low_threshold is None:
            low_threshold = self.canny_low
        if high_threshold is None:
            high_threshold = self.canny_high
        
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def sobel_edge_detection(self, image: np.ndarray, 
                            kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Apply Sobel edge detection
        
        Args:
            image: Input grayscale image
            kernel_size: Sobel kernel size (uses default if None)
            
        Returns:
            Edge magnitude map (0-255)
        """
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
        """
        Apply iterative smoothing + Canny for better edge detection
        
        Args:
            image: Input grayscale image
            iterations: Number of smoothing iterations (uses default if None)
            
        Returns:
            Combined edge map from all iterations
        """
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
        """
        Apply morphological closing to fill gaps in edges
        
        Args:
            edge_map: Binary edge map
            kernel_size: Size of structuring element
            
        Returns:
            Cleaned edge map
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
    
    def neighborhood_cleanup(self, edge_map: np.ndarray, threshold: int = 4) -> np.ndarray:
        """
        Clean edge map using neighborhood rule (majority voting)
        
        Args:
            edge_map: Binary edge map
            threshold: Minimum number of edge neighbors to keep pixel (out of 8)
            
        Returns:
            Cleaned edge map
        """
        # Pad the image to handle borders
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
        """
        Main function: Get clean edge map from input image
        
        Args:
            image: Input BGR or grayscale image
            method: Edge detection method ('canny', 'sobel', or 'iterative')
            cleanup_method: Cleanup method ('morphological' or 'neighborhood')
            
        Returns:
            Clean binary edge map
            
        Recommended defaults:
            - method='iterative' for complex objects
            - cleanup_method='morphological' for speed
            - Gaussian sigma=1.5, Canny thresholds=(50, 150)
        """
        # Convert to grayscale if needed
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


# Example usage and testing
if __name__ == "__main__":
    # CLI: process an image or folder directly
    import argparse
    import glob
    import os

    def _process_and_save(detector: EdgeDetector, path: str, out_dir: str, method: str, cleanup: str) -> bool:
        img = detector.load_image(path)
        if img is None:
            print(f"Skipping (can't read): {path}")
            return False

        edges = detector.get_clean_edges(img, method=method, cleanup_method=cleanup)

        # Ensure output dir exists
        os.makedirs(out_dir, exist_ok=True)

        # Make both arrays 3-channel so we can stack them for visualization
        gray = detector.to_grayscale(img)
        left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
        right = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges
        combined = np.hstack([left, right])

        base = os.path.splitext(os.path.basename(path))[0]
        out_file = os.path.join(out_dir, f"{base}_{method}_{cleanup}.png")
        ok = cv2.imwrite(out_file, combined)
        if ok:
            print(f"Saved: {out_file}")
        else:
            print(f"Failed to write: {out_file}")

        return ok

    parser = argparse.ArgumentParser(description="EdgeDetector demo - process image(s) and save visual outputs")
    parser.add_argument("-i", "--image", help="Path to single image to process")
    parser.add_argument("-d", "--dir", help="Directory to process recursively (jpg/png)")
    parser.add_argument("-m", "--method", choices=("canny", "sobel", "iterative"), default="iterative")
    parser.add_argument("-c", "--cleanup", choices=("morphological", "neighborhood", "none"), default="morphological")
    parser.add_argument("-o", "--out", default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'output')),
                        help="Output directory to save visualizations")
    parser.add_argument("--default-sample", action="store_true", help="Use a default sample from tests/test_images/BREAD_KNIFE")

    args = parser.parse_args()

    detector = EdgeDetector()
    processed = 0

    if args.default_sample and not args.image and not args.dir:
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sample_dir = os.path.join(proj_root, 'tests', 'test_images', 'BREAD_KNIFE')
        candidates = glob.glob(os.path.join(sample_dir, '*.jpg')) + glob.glob(os.path.join(sample_dir, '*.JPG'))
        if not candidates:
            print('No sample images found in', sample_dir)
        else:
            if _process_and_save(detector, candidates[0], args.out, args.method, args.cleanup):
                processed += 1

    if args.image:
        if _process_and_save(detector, args.image, args.out, args.method, args.cleanup):
            processed += 1

    if args.dir:
        for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG'):
            for path in glob.glob(os.path.join(args.dir, '**', ext), recursive=True):
                if _process_and_save(detector, path, args.out, args.method, args.cleanup):
                    processed += 1

    if processed == 0:
        print('\nNo images processed. Examples:')
        print('  python modules/preprocessing.py --default-sample')
        print('  python modules/preprocessing.py -i tests/test_images/BREAD_KNIFE/breadkniferaw1.jpg')
        print('  python modules/preprocessing.py -d tests/test_images/BREAD_KNIFE -m canny -c neighborhood')
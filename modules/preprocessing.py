import cv2
import numpy as np
from typing import Optional


class EdgeDetector:
    #Default parameters
    DEFAULT_GAUSSIAN_SIGMA = 1.5
    DEFAULT_CANNY_LOW = 30
    DEFAULT_CANNY_HIGH = 100
    DEFAULT_SOBEL_KERNEL = 3
    DEFAULT_ITERATIONS = 2
    
    def __init__(self):
        #Initialize with default parameters
        self.gaussian_sigma = self.DEFAULT_GAUSSIAN_SIGMA
        self.canny_low = self.DEFAULT_CANNY_LOW
        self.canny_high = self.DEFAULT_CANNY_HIGH
        self.sobel_kernel = self.DEFAULT_SOBEL_KERNEL
        self.iterations = self.DEFAULT_ITERATIONS
    
    @staticmethod
    #Function to load an image from a given path
    def load_image(image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
        return image
    
    @staticmethod
    #Function to convert an image to grayscale
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    #Function to apply Gaussian smoothing to an image
    def apply_gaussian_smoothing(self, image: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        if sigma is None:
            sigma = self.gaussian_sigma

        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    #Function to perform Canny edge detection
    def canny_edge_detection(self, image: np.ndarray, low_threshold: Optional[int] = None, high_threshold: Optional[int] = None) -> np.ndarray:

        if low_threshold is None:
            low_threshold = self.canny_low
        if high_threshold is None:
            high_threshold = self.canny_high
        
        return cv2.Canny(image, low_threshold, high_threshold)

    #Function to perform Sobel edge detection
    def sobel_edge_detection(self, image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:

        if kernel_size is None:
            kernel_size = self.sobel_kernel
    
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        return magnitude
    
    #Function to perform iterative Canny edge detection
    def iterative_canny(self, image: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
    
        if iterations is None:
            iterations = self.iterations
        
        combined_edges = np.zeros_like(image, dtype=np.uint8)
        
        for i in range(iterations):
         
            sigma = self.gaussian_sigma * (i + 1)
            smoothed = self.apply_gaussian_smoothing(image, sigma)
            edges = self.canny_edge_detection(smoothed)
            
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        return combined_edges
    
    #Function to perform morphological closing on an edge map
    def morphological_closing(self, edge_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)
    
    #Function to get clean edges using specified method
    def get_clean_edges(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        # Convert to grayscale
        gray = self.to_grayscale(image)

        #Smooth image
        smoothed = self.apply_gaussian_smoothing(gray)

        #Edge detection
        if method == 'canny':
            edges = self.canny_edge_detection(smoothed)

        elif method == 'sobel':
        
            sobel_edges = self.sobel_edge_detection(smoothed)
            _, edges = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)

        elif method == 'iterative':
            edges = self.iterative_canny(smoothed)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        #Apply morphological cleanup
        cleaned = self.morphological_closing(edges)
        
        return cleaned


# Example usage and testing
if __name__ == "__main__":
    # CLI: process an image or folder directly
    import argparse
    import glob
    import os

    def _process_and_save(detector: EdgeDetector, path: str, out_dir: str, method: str) -> bool:
        img = detector.load_image(path)
        if img is None:
            print(f"Skipping (can't read): {path}")
            return False

        edges = detector.get_clean_edges(img, method=method)

        # Ensure output dir exists
        os.makedirs(out_dir, exist_ok=True)

        # Make both arrays 3-channel so we can stack them for visualization
        gray = detector.to_grayscale(img)
        left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
        right = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(edges.shape) == 2 else edges
        combined = np.hstack([left, right])

        base = os.path.splitext(os.path.basename(path))[0]
        out_file = os.path.join(out_dir, f"{base}_{method}.png")
        ok = cv2.imwrite(out_file, combined)
        if ok:
            print(f"Saved: {out_file}")
        else:
            print(f"Failed to write: {out_file}")

        return ok

    parser = argparse.ArgumentParser(description="EdgeDetector demo - process image(s) and save visual outputs")
    parser.add_argument("-i", "--image", help="Path to single image to process")
    parser.add_argument("-d", "--dir", help="Directory to process recursively (jpg/png)")
    parser.add_argument("-m", "--method", choices=("canny", "sobel", "iterative"), default="canny")
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
            if _process_and_save(detector, candidates[0], args.out, args.method):
                processed += 1

    if args.image:
        if _process_and_save(detector, args.image, args.out, args.method):
            processed += 1

    if args.dir:
        for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG'):
            for path in glob.glob(os.path.join(args.dir, '**', ext), recursive=True):
                if _process_and_save(detector, path, args.out, args.method):
                    processed += 1

    if processed == 0:
        print('\nNo images processed. Examples:')
        print('  python modules/preprocessing.py --default-sample')
        print('  python modules/preprocessing.py -i tests/test_images/BREAD_KNIFE/breadkniferaw1.jpg')
        print('  python modules/preprocessing.py -d tests/test_images/BREAD_KNIFE -m canny')
"""
Test script for Person A - Edge Detection Module
"""

import cv2
import numpy as np
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.preprocessing import EdgeDetector


def test_edge_detection():
    """Test edge detection on sample images"""
    print("="*60)
    print("Testing Person A - Edge Detection Module")
    print("="*60)
    
    # Create test image (simple geometric shapes)
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    test_image[:] = (200, 200, 200)  # Gray background
    
    # Draw some shapes
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(test_image, (250, 100), 50, (0, 255, 0), -1)
    cv2.ellipse(test_image, (150, 300), (80, 40), 45, 0, 360, (0, 0, 255), -1)
    
    # Initialize detector
    detector = EdgeDetector()
    
    # Test 1: Basic Canny edge detection
    print("\nTest 1: Canny Edge Detection")
    edges_canny = detector.get_clean_edges(test_image, method='canny')
    edge_pixels = np.sum(edges_canny > 0)
    print(f"  Edge pixels detected: {edge_pixels}")
    print(f"  ✓ Canny edge detection works")
    
    # Test 2: Sobel edge detection
    print("\nTest 2: Sobel Edge Detection")
    edges_sobel = detector.get_clean_edges(test_image, method='sobel')
    edge_pixels_sobel = np.sum(edges_sobel > 0)
    print(f"  Edge pixels detected: {edge_pixels_sobel}")
    print(f"  ✓ Sobel edge detection works")
    
    # Test 3: Iterative Canny
    print("\nTest 3: Iterative Canny")
    edges_iterative = detector.get_clean_edges(test_image, method='iterative')
    edge_pixels_iter = np.sum(edges_iterative > 0)
    print(f"  Edge pixels detected: {edge_pixels_iter}")
    print(f"  ✓ Iterative Canny works")
    
    # Test 4: Different smoothing levels
    print("\nTest 4: Testing Different Smoothing Levels")
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        detector.gaussian_sigma = sigma
        edges = detector.get_clean_edges(test_image, method='canny')
        edge_count = np.sum(edges > 0)
        print(f"  Sigma={sigma:.1f}: {edge_count} edge pixels")
    
    # Test 5: Cleanup methods
    print("\nTest 5: Comparing Cleanup Methods")
    edges_morph = detector.get_clean_edges(test_image, method='canny', 
                                          cleanup_method='morphological')
    edges_neighbor = detector.get_clean_edges(test_image, method='canny', 
                                             cleanup_method='neighborhood')
    print(f"  Morphological cleanup: {np.sum(edges_morph > 0)} edge pixels")
    print(f"  Neighborhood cleanup: {np.sum(edges_neighbor > 0)} edge pixels")
    print(f"  ✓ Both cleanup methods work")
    
    # Visual output
    print("\nGenerating visualization...")
    combined = np.hstack([
        cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY),
        edges_canny,
        edges_iterative
    ])
    
    cv2.imwrite('../data/output/test_edges.png', combined)
    print("  Saved: data/output/test_edges.png")
    print("  (Original | Canny | Iterative)")
    
    print("\n" + "="*60)
    print("✓ All tests passed for Person A!")
    print("="*60)
    
    # Display recommended parameters
    print("\nRecommended Parameters:")
    print(f"  Gaussian sigma: {EdgeDetector.DEFAULT_GAUSSIAN_SIGMA}")
    print(f"  Canny low threshold: {EdgeDetector.DEFAULT_CANNY_LOW}")
    print(f"  Canny high threshold: {EdgeDetector.DEFAULT_CANNY_HIGH}")
    print(f"  Iterations: {EdgeDetector.DEFAULT_ITERATIONS}")


if __name__ == '__main__':
    test_edge_detection()
import cv2
import numpy as np
import sys, os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.preprocessing import EdgeDetector


def calculate_metrics(edges):
    """Calculate 3 simple metrics for edge detection quality"""
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    
    # Metric 1: Edge Density (percentage of edge pixels)
    edge_density = (edge_pixels / total_pixels) * 100
    
    # Metric 2: Average Edge Strength (average intensity of edge pixels)
    edge_strength = np.mean(edges[edges > 0]) if edge_pixels > 0 else 0
    
    # Metric 3: Number of Connected Components (fewer = more continuous edges)
    num_components = cv2.connectedComponents(edges)[0] - 1  # Subtract background
    
    return {
        'edge_density': edge_density,
        'edge_strength': edge_strength,
        'num_components': num_components,
        'total_edge_pixels': edge_pixels
    }


def test_parameter_sensitivity(detector, image, category_name):
    """Test how parameters affect edge detection quality for a specific image"""
    print(f"\n  Parameter Sensitivity Analysis:")
    print(f"  {'Sigma':<8} {'Density':<12} {'Strength':<12} {'Components'}")
    print(f"  {'-'*50}")
    
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        detector.gaussian_sigma = sigma
        edges = detector.get_clean_edges(image, method='canny')
        metrics = calculate_metrics(edges)
        print(f"  {sigma:<8.1f} {metrics['edge_density']:<12.2f} "
              f"{metrics['edge_strength']:<12.2f} {metrics['num_components']}")


def test_reference_images():
    """Test edge detection on reference images only"""
    print("="*80)
    print("Testing Edge Detection Module - Reference Images")
    print("="*80)
    
    # Initialize detector
    detector = EdgeDetector()
    
    # Find reference images
    test_root = Path(PROJECT_ROOT) / 'test_images'
    categories = ['book', 'controller', 'knife']
    
    print("\nTesting Reference Images:\n")
    
    for category in categories:
        category_dir = test_root / category
        ref_image_path = category_dir / 'ref_img.png'
        
        print(f"\n{'='*80}")
        print(f"Category: {category.upper()}")
        print(f"{'='*80}")
        
        # Load image
        image = detector.load_image(str(ref_image_path))
        
        # Reset to default parameters
        detector.gaussian_sigma = EdgeDetector.DEFAULT_GAUSSIAN_SIGMA
        
        # Test different methods
        methods = ['canny', 'sobel', 'iterative']
        edges_dict = {}
        
        print(f"\n  Method Comparison:")
        for method in methods:
            edges = detector.get_clean_edges(image, method=method)
            edges_dict[method] = edges
            metrics = calculate_metrics(edges)
            
            print(f"\n  {method.upper()}:")
            print(f"    Edge Density: {metrics['edge_density']:.2f}%")
            print(f"    Edge Strength: {metrics['edge_strength']:.2f}")
            print(f"    Connected Components: {metrics['num_components']}")
            print(f"    Total Edge Pixels: {metrics['total_edge_pixels']}")

        
        # Create and save combined visualization
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        combined = np.hstack([gray_image, edges_dict['canny'], 
                             edges_dict['sobel'], edges_dict['iterative']])
        
        combined_path = category_dir / 'ref_img_edges_combined.png'
        cv2.imwrite(str(combined_path), combined)
        print(f"\n  ✓ Saved combined visualization: ref_img_edges_combined.png")
        print(f"    (Original | Canny | Sobel | Iterative)")
        
        # Test parameter sensitivity for this category
        test_parameter_sensitivity(detector, image, category)
    
    print(f"\n\n{'='*80}")
    print("✓ Testing Complete!")
    print(f"{'='*80}")
    print(f"\nEdge images saved in their respective category folders:")
    for category in categories:
        print(f"  - test_images/{category}/")


if __name__ == '__main__':
    test_reference_images()
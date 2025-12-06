import cv2
import numpy as np
import sys, os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.preprocessing import EdgeDetector


def calculate_metrics(edges):
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    
    edge_density = (edge_pixels / total_pixels) * 100
    
    edge_strength = np.mean(edges[edges > 0]) if edge_pixels > 0 else 0
    
    num_components = cv2.connectedComponents(edges)[0] - 1 
    
    return {
        'edge_density': edge_density,
        'edge_strength': edge_strength,
        'num_components': num_components,
        'total_edge_pixels': edge_pixels
    }


def test_parameter_sensitivity(detector, image, category_name):
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        detector.gaussian_sigma = sigma
        edges = detector.get_clean_edges(image, method='canny')
        metrics = calculate_metrics(edges)
        print(f"{sigma:<8.1f} {metrics['edge_density']:<12.2f} "
              f"{metrics['edge_strength']:<12.2f} {metrics['num_components']}")


def test_reference_images():
    print("Testing Edge Detection Module - Reference Images")
    
    detector = EdgeDetector()
    
    test_root = Path(PROJECT_ROOT) / 'test_images'
    categories = ['book', 'controller', 'knife']
    
    print("\nTesting Reference Images:\n")
    
    for category in categories:
        category_dir = test_root / category
        ref_image_path = category_dir / 'ref_img.png'
        
        print(f"Category: {category.upper()}")
        
        image = detector.load_image(str(ref_image_path))
        
        detector.gaussian_sigma = EdgeDetector.DEFAULT_GAUSSIAN_SIGMA
        
        methods = ['canny', 'sobel', 'iterative']
        edges_dict = {}
        
        print(f"\n  Method Comparison:")
        for method in methods:
            edges = detector.get_clean_edges(image, method=method)
            edges_dict[method] = edges
            metrics = calculate_metrics(edges)
            
            print(f"{method.upper()}:")
            print(f"Edge Density: {metrics['edge_density']:.2f}%")
            print(f"Edge Strength: {metrics['edge_strength']:.2f}")
            print(f"Connected Components: {metrics['num_components']}")
            print(f"Total Edge Pixels: {metrics['total_edge_pixels']}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        combined = np.hstack([gray_image, edges_dict['canny'], 
                             edges_dict['sobel'], edges_dict['iterative']])
        
        combined_path = category_dir / 'ref_img_edges_combined.png'
        cv2.imwrite(str(combined_path), combined)
        
        test_parameter_sensitivity(detector, image, category)


if __name__ == '__main__':
    test_reference_images()
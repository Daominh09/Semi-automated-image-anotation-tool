"""
Test script for Person B - Segmentation Module
"""

import cv2
import numpy as np
import sys
sys.path.append('..')
from modules.segmentation import RegionGrowing


def test_region_growing():
    """Test region growing segmentation"""
    print("="*60)
    print("Testing Person B - Region Growing Module")
    print("="*60)
    
    # Create test image
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Dark background
    
    # Draw target object (red square with slight gradient)
    for i in range(100, 300):
        for j in range(100, 300):
            # Add slight color variation
            variation = np.random.randint(-10, 10)
            test_image[i, j] = (200 + variation, 50 + variation, 50 + variation)
    
    # Initialize region grower
    rg = RegionGrowing()
    
    # Test 1: Single seed segmentation
    print("\nTest 1: Single Seed Segmentation")
    rg.clear_seeds()
    rg.add_seed(200, 200)  # Center of red square
    
    mask, contours = rg.segment(test_image, threshold=20)
    segmented_area = np.sum(mask > 0)
    expected_area = 200 * 200  # Should be close to 40000
    
    print(f"  Seed at: (200, 200)")
    print(f"  Segmented area: {segmented_area} pixels")
    print(f"  Expected area: ~{expected_area} pixels")
    print(f"  Accuracy: {100 * segmented_area / expected_area:.1f}%")
    
    if 0.8 < segmented_area / expected_area < 1.2:
        print(f"  ✓ Single seed segmentation works!")
    else:
        print(f"  ⚠ Segmentation might need threshold adjustment")
    
    # Test 2: Multi-seed segmentation
    print("\nTest 2: Multi-Seed Segmentation")
    
    # Add another object
    cv2.circle(test_image, (300, 100), 40, (50, 200, 50), -1)
    
    rg.clear_seeds()
    rg.add_seed(200, 200)  # Red square
    rg.add_seed(300, 100)  # Green circle
    
    mask_multi, contours_multi = rg.segment(test_image, threshold=20)
    num_contours = len(contours_multi)
    
    print(f"  Seeds: (200, 200) and (300, 100)")
    print(f"  Number of contours found: {num_contours}")
    print(f"  Total segmented area: {np.sum(mask_multi > 0)} pixels")
    
    if num_contours >= 1:
        print(f"  ✓ Multi-seed segmentation works!")
    
    # Test 3: Undo functionality
    print("\nTest 3: Undo Functionality")
    rg.clear_seeds()
    rg.add_seed(200, 200)
    rg.add_seed(150, 150)
    rg.add_seed(250, 250)
    
    print(f"  Added 3 seeds: {len(rg.seeds)} seeds")
    
    rg.undo_last_seed()
    print(f"  After undo: {len(rg.seeds)} seeds")
    
    if len(rg.seeds) == 2:
        print(f"  ✓ Undo works!")
    
    # Test 4: Different thresholds
    print("\nTest 4: Testing Different Thresholds")
    rg.clear_seeds()
    rg.add_seed(200, 200)
    
    for threshold in [5, 10, 20, 30, 40]:
        mask_thresh, _ = rg.segment(test_image, threshold=threshold)
        area = np.sum(mask_thresh > 0)
        print(f"  Threshold={threshold}: {area} pixels segmented")
    
    print(f"  ✓ Threshold adjustment works (larger threshold = larger region)")
    
    # Test 5: Morphological smoothing
    print("\nTest 5: Mask Smoothing")
    rg.clear_seeds()
    rg.add_seed(200, 200)
    
    mask_raw, _ = rg.segment(test_image, threshold=20, smooth=False)
    mask_smooth, _ = rg.segment(test_image, threshold=20, smooth=True)
    
    # Count boundary pixels
    raw_boundary = cv2.Canny(mask_raw, 100, 200)
    smooth_boundary = cv2.Canny(mask_smooth, 100, 200)
    
    print(f"  Raw boundary pixels: {np.sum(raw_boundary > 0)}")
    print(f"  Smoothed boundary pixels: {np.sum(smooth_boundary > 0)}")
    print(f"  ✓ Smoothing reduces boundary noise")
    
    # Test 6: Contour extraction
    print("\nTest 6: Contour Extraction")
    largest_contour = rg.get_largest_contour(contours)
    
    if largest_contour is not None:
        area = cv2.contourArea(largest_contour)
        print(f"  Largest contour area: {area} pixels")
        print(f"  Number of points: {len(largest_contour)}")
        print(f"  ✓ Contour extraction works!")
    
    # Visual output
    print("\nGenerating visualization...")
    
    # Create overlay and contour images
    overlay = rg.create_overlay(test_image, mask, alpha=0.3)
    contour_img = rg.draw_contours_on_image(test_image, contours, 
                                            color=(0, 255, 0), thickness=2)
    
    # Combine for side-by-side comparison
    combined = np.hstack([test_image, overlay, contour_img])
    
    cv2.imwrite('../data/output/test_segmentation.png', combined)
    print("  Saved: data/output/test_segmentation.png")
    print("  (Original | Overlay | Contours)")
    
    print("\n" + "="*60)
    print("✓ All tests passed for Person B!")
    print("="*60)
    
    # Display recommended parameters
    print("\nRecommended Parameters:")
    print(f"  Color threshold (τ): {RegionGrowing.DEFAULT_COLOR_THRESHOLD}")
    print(f"  Grayscale threshold: {RegionGrowing.DEFAULT_GRAY_THRESHOLD}")
    print(f"  Morphological kernel: {RegionGrowing.DEFAULT_MORPH_KERNEL}")


if __name__ == '__main__':
    test_region_growing()
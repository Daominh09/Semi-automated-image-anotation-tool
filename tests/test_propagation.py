import cv2
import numpy as np
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.propagation import AnnotationPropagator


def test_category(propagator, category_path, category_name, output_base_dir):
    ref_img_path = os.path.join(category_path, "ref_img.png")
    ref_mask_path = os.path.join(category_path, "ref_img_mask.png")
    
    if not os.path.exists(ref_img_path):
        print(f"Warning: {ref_img_path} not found, skipping category {category_name}")
        return
    if not os.path.exists(ref_mask_path):
        print(f"Warning: {ref_mask_path} not found, skipping category {category_name}")
        return
    
    ref_frame = cv2.imread(ref_img_path)
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_frame is None or ref_mask is None:
        print(f"Error loading reference files for {category_name}")
        return
    
    h, w = ref_frame.shape[:2]
    max_edge = max(h, w)
    
    if max_edge > 500:
        scale = 500 / max_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        ref_frame = cv2.resize(ref_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ref_mask = cv2.resize(ref_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"Reference resized from {w}x{h} to {new_w}x{new_h}")
    else:
        print(f"Reference size {w}x{h} - no resizing needed")
    
    print(f"Testing category: {category_name}")
    print(f"Reference image: {ref_img_path}")
    print(f"Reference mask: {ref_mask_path}")
    
    category_output_dir = os.path.join(output_base_dir, category_name)
    os.makedirs(category_output_dir, exist_ok=True)
    
    results_summary = []
    
    for i in range(1, 6):
        img_filename = f"img{i}.png"
        img_path = os.path.join(category_path, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        next_frame = cv2.imread(img_path)
        if next_frame is None:
            print(f"Error loading {img_path}, skipping")
            continue
        
        h_next, w_next = next_frame.shape[:2]
        max_edge_next = max(h_next, w_next)
        
        if max_edge_next > 500:
            scale_next = 500 / max_edge_next
            new_w_next = int(w_next * scale_next)
            new_h_next = int(h_next * scale_next)
            next_frame = cv2.resize(next_frame, (new_w_next, new_h_next), interpolation=cv2.INTER_AREA)
        
        propagated_mask, metrics = propagator.propagate_annotation(
            ref_frame, ref_mask, next_frame
        )
        
        print(f"Matches: {metrics['matches']}")
        print(f"Candidate generated: {metrics['candidates_generated']}")
        print(f"Transform success: {metrics['transform_success']}")
        
        overlay = next_frame.copy()
        overlay[propagated_mask > 0] = [0, 255, 0]
        result = cv2.addWeighted(next_frame, 0.7, overlay, 0.3, 0)
        
        mask_output_path = os.path.join(category_output_dir, f"mask_{i}.png")
        result_output_path = os.path.join(category_output_dir, f"result_{i}.png")
        
        cv2.imwrite(mask_output_path, propagated_mask)
        cv2.imwrite(result_output_path, result)
        
        print(f"Saved: {mask_output_path}")
        print(f"Saved: {result_output_path}")
        
        # Store results for summary
        results_summary.append({
            'image': img_filename,
            'matches': metrics['matches'],
            'success': metrics['transform_success']
        })
    
    return results_summary


if __name__ == "__main__":
    # Initialize propagator
    propagator = AnnotationPropagator()
    print("SIFT Annotation Propagator initialized")
    
    # Define test images base directory
    test_images_dir = "test_images"
    
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory '{test_images_dir}' not found")
        sys.exit(1)
    
    categories = ["book", "controller", "knife"]
    
    output_base_dir = "propagation_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_results = {}
    
    for category in categories:
        category_path = os.path.join(test_images_dir, category)
        
        if not os.path.exists(category_path):
            print(f"\nWarning: Category directory '{category_path}' not found, skipping")
            continue
        
        results = test_category(propagator, category_path, category, output_base_dir)
        if results:
            all_results[category] = results
    
    print(f"Final summary - All categories")
    
    for category, results in all_results.items():
        total_images = len(results)
        successful = sum(1 for r in results if r['success'])
        avg_matches = sum(r['matches'] for r in results) / total_images if total_images > 0 else 0
        
        print(f"{category.upper()}:")
        print(f"Total images: {total_images}")
        print(f"Successful: {successful}/{total_images} ({100*successful/total_images:.1f}%)")
        print(f"Avg matches: {avg_matches:.1f}")
    
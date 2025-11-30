import cv2
import numpy as np
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from modules.propagation import AnnotationPropagator


if __name__ == "__main__":
    import sys
    import os

    # Initialize propagator
    propagator = AnnotationPropagator()
    print("SIFT Annotation Propagator initialized")

    # Check command line args
    if len(sys.argv) < 4:
        print("Usage: python script.py <image1_path> <image2_path> <mask_path> [--visualize|-v]")
        print("Example: python script.py prev.jpg next.jpg mask.png --visualize")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    mask_path = sys.argv[3]
    visualize_mode = "--visualize" in sys.argv or "-v" in sys.argv

    # Load images
    prev_frame = cv2.imread(img1_path)
    next_frame = cv2.imread(img2_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if prev_frame is None:
        print(f"Error: Could not load image {img1_path}")
        sys.exit(1)
    if next_frame is None:
        print(f"Error: Could not load image {img2_path}")
        sys.exit(1)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        sys.exit(1)

    print(f"Loaded images: {img1_path}, {img2_path}")
    print(f"Loaded mask: {mask_path}")
    print(
        f"Image shapes: prev={prev_frame.shape}, "
        f"next={next_frame.shape}, mask={mask.shape}"
    )

    if visualize_mode:
        print("\n=== Running visualization pipeline ===")
        output_dir = "visualization_output"
        os.makedirs(output_dir, exist_ok=True)

        results = propagator.visualize_propagation_pipeline(
            prev_frame, mask, next_frame, save_dir=output_dir
        )

        print("\n=== Visualization complete ===")
        print(f"Output directory: {output_dir}")
        print(f"Metrics: {results.get('metrics', {})}")

        # Show windows if images exist
        if results.get("keypoints_source"):
            img = cv2.imread(results["keypoints_source"])
            cv2.imshow("1. Source Keypoints", img)

        if results.get("keypoints_target"):
            img = cv2.imread(results["keypoints_target"])
            cv2.imshow("2. Target Keypoints", img)

        if results.get("matches"):
            img = cv2.imread(results["matches"])
            cv2.imshow("3. Feature Matches", img)

        if results.get("best_points"):
            img = cv2.imread(results["best_points"])
            cv2.imshow("3b. Inlier Points", img)

        if results.get("result"):
            img = cv2.imread(results["result"])
            cv2.imshow("4. Propagation Result", img)

        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # Simple propagation-only path
        propagated_mask, metrics = propagator.propagate_annotation(
            prev_frame, mask, next_frame
        )

        print("\nPropagation metrics:")
        print(f"  Matches: {metrics['matches']}")
        print(f"  Inliers: {metrics['inliers']}")
        print(f"  Transform success: {metrics['transform_success']}")

        overlay = next_frame.copy()
        print(f"Overlay shape: {overlay.shape}")
        print(f"Propagated mask shape: {propagated_mask.shape}")
        overlay[propagated_mask > 0] = [0, 255, 0]
        result = cv2.addWeighted(next_frame, 0.7, overlay, 0.3, 0)

        cv2.putText(
            result,
            f"Matches: {metrics['matches']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result,
            f"Success: {metrics['transform_success']}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Save outputs
        output_mask_path = "propagated_mask.png"
        output_result_path = "result_visualization.png"
        cv2.imwrite(output_mask_path, propagated_mask)
        cv2.imwrite(output_result_path, result)
        print(f"\nPropagated mask saved: {output_mask_path}")
        print(f"Visualization saved: {output_result_path}")

        # Show results
        cv2.imshow("Propagated Annotation", result)
        cv2.imshow("Propagated Mask", propagated_mask)
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""
Evaluation Metrics Module
Used by all team members for measuring annotation accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class AnnotationMetrics:
    """Calculate evaluation metrics for annotation quality"""
    
    @staticmethod
    def intersection_over_union(pred_mask: np.ndarray, 
                               gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) / Jaccard Index
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score (0-1), higher is better
            
        Formula: IoU = |A ∩ B| / |A ∪ B|
        """
        # Ensure binary masks
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    @staticmethod
    def dice_coefficient(pred_mask: np.ndarray, 
                        gt_mask: np.ndarray) -> float:
        """
        Calculate Dice Coefficient (F1 Score)
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Dice score (0-1), higher is better
            
        Formula: Dice = 2|A ∩ B| / (|A| + |B|)
        """
        # Ensure binary masks
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate intersection
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        
        # Calculate areas
        pred_area = pred_binary.sum()
        gt_area = gt_binary.sum()
        
        if pred_area + gt_area == 0:
            return 0.0
        
        dice = (2.0 * intersection) / (pred_area + gt_area)
        return float(dice)
    
    @staticmethod
    def pixel_accuracy(pred_mask: np.ndarray, 
                      gt_mask: np.ndarray) -> float:
        """
        Calculate pixel-wise accuracy
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Accuracy score (0-1)
        """
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        correct = (pred_binary == gt_binary).sum()
        total = pred_binary.size
        
        return float(correct / total)
    
    @staticmethod
    def precision_recall(pred_mask: np.ndarray, 
                        gt_mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate precision and recall
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Tuple of (precision, recall)
        """
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        true_positive = np.logical_and(pred_binary, gt_binary).sum()
        false_positive = np.logical_and(pred_binary, np.logical_not(gt_binary)).sum()
        false_negative = np.logical_and(np.logical_not(pred_binary), gt_binary).sum()
        
        # Precision: TP / (TP + FP)
        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)
        
        # Recall: TP / (TP + FN)
        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)
        
        return float(precision), float(recall)
    
    @staticmethod
    def boundary_f1(pred_mask: np.ndarray, 
                   gt_mask: np.ndarray,
                   threshold: float = 2.0) -> float:
        """
        Calculate Boundary F1 score (measures boundary accuracy)
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            threshold: Distance threshold for boundary matching (pixels)
            
        Returns:
            Boundary F1 score (0-1)
        """
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Extract boundaries using Canny
        pred_boundary = cv2.Canny(pred_binary * 255, 100, 200)
        gt_boundary = cv2.Canny(gt_binary * 255, 100, 200)
        
        # Get boundary coordinates
        pred_coords = np.column_stack(np.where(pred_boundary > 0))
        gt_coords = np.column_stack(np.where(gt_boundary > 0))
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return 0.0
        
        # Calculate distances
        from scipy.spatial.distance import cdist
        
        # Distance from predicted to ground truth
        dist_pred_to_gt = cdist(pred_coords, gt_coords, metric='euclidean')
        min_dist_pred = np.min(dist_pred_to_gt, axis=1)
        
        # Distance from ground truth to predicted
        dist_gt_to_pred = cdist(gt_coords, pred_coords, metric='euclidean')
        min_dist_gt = np.min(dist_gt_to_pred, axis=1)
        
        # Count matches within threshold
        pred_matches = (min_dist_pred <= threshold).sum()
        gt_matches = (min_dist_gt <= threshold).sum()
        
        # Calculate precision and recall
        precision = pred_matches / len(pred_coords)
        recall = gt_matches / len(gt_coords)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)
    
    @staticmethod
    def evaluate_all(pred_mask: np.ndarray, 
                    gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics at once
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        metrics['iou'] = AnnotationMetrics.intersection_over_union(pred_mask, gt_mask)
        metrics['dice'] = AnnotationMetrics.dice_coefficient(pred_mask, gt_mask)
        metrics['pixel_accuracy'] = AnnotationMetrics.pixel_accuracy(pred_mask, gt_mask)
        
        precision, recall = AnnotationMetrics.precision_recall(pred_mask, gt_mask)
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # F1 from precision and recall
        if precision + recall > 0:
            metrics['f1'] = 2 * precision * recall / (precision + recall)
        else:
            metrics['f1'] = 0.0
        
        # Boundary F1 requires scipy, so make it optional
        try:
            metrics['boundary_f1'] = AnnotationMetrics.boundary_f1(pred_mask, gt_mask)
        except ImportError:
            metrics['boundary_f1'] = None
        
        return metrics


class ResultsLogger:
    """Log and save evaluation results"""
    
    def __init__(self):
        """Initialize empty results storage"""
        self.results = []
    
    def add_result(self, image_name: str, 
                  method: str,
                  metrics: Dict[str, float],
                  time_taken: float = None):
        """
        Add a result entry
        
        Args:
            image_name: Name/ID of the image
            method: Method used (e.g., 'manual', 'auto', 'propagated')
            metrics: Dictionary of metric scores
            time_taken: Time taken in seconds (optional)
        """
        result = {
            'image': image_name,
            'method': method,
            'time_seconds': time_taken,
            **metrics
        }
        self.results.append(result)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get results as pandas DataFrame
        
        Returns:
            DataFrame with all results
        """
        return pd.DataFrame(self.results)
    
    def save_to_excel(self, filepath: str):
        """
        Save results to Excel file
        
        Args:
            filepath: Path to save Excel file
        """
        df = self.get_dataframe()
        df.to_excel(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def save_to_csv(self, filepath: str):
        """
        Save results to CSV file
        
        Args:
            filepath: Path to save CSV file
        """
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary statistics"""
        df = self.get_dataframe()
        
        if len(df) == 0:
            print("No results to summarize")
            return
        
        print("\n" + "="*60)
        print("ANNOTATION ACCURACY SUMMARY")
        print("="*60)
        
        # Group by method
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            print(f"\nMethod: {method.upper()}")
            print("-" * 40)
            
            if 'iou' in method_data.columns:
                print(f"  IoU:           {method_data['iou'].mean():.4f} ± {method_data['iou'].std():.4f}")
            
            if 'dice' in method_data.columns:
                print(f"  Dice:          {method_data['dice'].mean():.4f} ± {method_data['dice'].std():.4f}")
            
            if 'pixel_accuracy' in method_data.columns:
                print(f"  Pixel Acc:     {method_data['pixel_accuracy'].mean():.4f} ± {method_data['pixel_accuracy'].std():.4f}")
            
            if 'f1' in method_data.columns:
                print(f"  F1 Score:      {method_data['f1'].mean():.4f} ± {method_data['f1'].std():.4f}")
            
            if 'time_seconds' in method_data.columns and method_data['time_seconds'].notna().any():
                print(f"  Avg Time:      {method_data['time_seconds'].mean():.2f}s")
        
        print("\n" + "="*60)


# Example usage and testing
if __name__ == "__main__":
    print("AnnotationMetrics module loaded successfully!")
    print("\nMetrics explanation:")
    print("  IoU (Intersection over Union):  Overlap / Union")
    print("  Dice Coefficient:               2 * Overlap / (Area1 + Area2)")
    print("  Target values:                  IoU > 0.7, Dice > 0.8")
    print("\nHigher scores = better accuracy")
    
    # Example: Create dummy masks and calculate metrics
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[25:75, 25:75] = 255  # 50x50 square
    
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:70, 20:70] = 255    # 50x50 square, slightly offset
    
    metrics = AnnotationMetrics.evaluate_all(pred, gt)
    print("\nExample metrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
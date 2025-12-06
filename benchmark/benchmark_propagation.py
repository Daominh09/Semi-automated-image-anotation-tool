import os
import cv2
import numpy as np
import csv

def binarize(mask):
    return mask > 0

def segmentation_metrics(pred_mask, gt_mask):
    pred = binarize(pred_mask)
    gt   = binarize(gt_mask)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    pred_sum     = pred.sum()
    gt_sum       = gt.sum()

    iou  = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0

    tp = intersection
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

def evaluate_propagation_results(root_dir="propagation_results",
                                 csv_path="propagation_results_metrics.csv"):
    object_names = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    results = []

    for obj in object_names:
        base = os.path.join(root_dir, obj)

        for i in range(1, 6):
            pred_path = os.path.join(base, f"mask_{i}.png")
            gt_path   = os.path.join(base, f"mask_{i}_gt.png")

            if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                continue

            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_mask   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if pred_mask is None or gt_mask is None:
                continue

            h, w = gt_mask.shape[:2]
            max_edge = max(h, w)

            if max_edge > 500:
                scale = 500 / max_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                gt_mask = cv2.resize(gt_mask, (new_w, new_h),
                                     interpolation=cv2.INTER_NEAREST)
                
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            metrics = segmentation_metrics(pred_mask, gt_mask)
            results.append({
                "object": obj,
                "mask_id": i,
                **metrics
            })

    if results:
        metric_keys = ["IoU", "Dice", "Accuracy", "Precision", "Recall", "F1"]
        mean_metrics = {
            k: float(np.mean([r[k] for r in results]))
            for k in metric_keys
        }
        results.append({
            "object": "mean",
            "mask_id": 0,
            **mean_metrics
        })

    fieldnames = ["object", "mask_id",
                  "IoU", "Dice", "Accuracy", "Precision", "Recall", "F1"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"object": r["object"]}
            for k in fieldnames[1:]:
                row[k] = f"{r[k]:.4f}"
            writer.writerow(row)

if __name__ == "__main__":
    evaluate_propagation_results("propagation_results",
                                 "benchmark/propagation_results.csv")

import numpy as np
import cv2
from typing import Dict, Any
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

def evaluate_segmentation(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    true_positive = intersection
    false_positive = np.logical_and(pred, np.logical_not(gt)).sum()
    false_negative = np.logical_and(np.logical_not(pred), gt).sum()

    iou = intersection / union if union > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    
metrics = evaluate_segmentation(pred_mask, ground_truth_mask)
print(metrics)

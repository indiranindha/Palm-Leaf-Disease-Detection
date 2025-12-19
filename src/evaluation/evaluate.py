import torch
import torch.nn.functional as F
from src.evaluation.metrics import compute_metrics

def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    acc, precision, recall, f1, cm = compute_metrics(
        all_labels, all_preds
    )

    avg_conf = sum(all_confidences) / len(all_confidences)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_confidence": avg_conf,
        "confusion_matrix": cm
    }

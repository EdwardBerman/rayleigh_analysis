import torch
import torch.nn.functional as F

import numpy as np

from sklearn.metrics import f1_score, average_precision_score

def node_level_accuracy(node_logits_batch, labels_batch):
    """
    Compute the accuracy of node classification.

    Parameters:
    - node_logits_batch: Tensor of shape (num_nodes, num_classes), raw output logits from the model.
    - labels_batch: Tensor of shape (num_nodes,), true labels for each node.

    Returns:
    - accuracy: Float, the accuracy of the predictions.
    """

    preds = F.log_softmax(node_logits_batch, dim=1)

    _, predicted_classes = torch.max(preds, dim=1)
    correct_predictions = (predicted_classes == labels_batch).sum().item()

    total_predictions = labels_batch.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy

@torch.no_grad()
def eval_F1(node_logits_batch, true):

    preds = F.log_softmax(node_logits_batch, dim=1)

    _, predicted_classes = torch.max(preds, dim=1)

    pred_np = predicted_classes.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    f1 = f1_score(true_np, pred_np, average='macro')
    return f1

def graph_level_average_precision(y_pred, y_true):

    ap_list = []

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)

def graph_level_accuracy(pred, true):
    """
    pred: [B, C] raw logits
    true: [B, C] or [B, 1, C] with 0/1 labels
    """
    true = true.float()
    if true.ndim > 2:
        true = true.view(true.size(0), -1)

    if true.shape != pred.shape:
        true = true.view_as(pred)

    probs = torch.sigmoid(pred)
    preds = (probs > 0.5).float()

    correct = (preds == true).float().mean(dim=1)
    return correct.mean()

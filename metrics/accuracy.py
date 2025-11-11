import torch
import torch.nn.functional as F

import numpy as np

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

def eval_F1(pred, true):

    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    if pred.ndim > 1:
        pred_labels = np.argmax(pred, axis=-1)
    else:
        pred_labels = (pred > 0).astype(int)
    
    seq_ref = [set([t]) for t in true]
    seq_pred = [set([p]) for p in pred_labels]
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for label, prediction in zip(seq_ref, seq_pred):
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)
        
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    return np.average(f1_list)

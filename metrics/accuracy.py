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

def eval_F1(pred, true, threshold=0.5):
    # '''
    #     compute F1 score averaged over samples
    # '''

    if pred.dtype == torch.float:
        pred_binary = (torch.sigmoid(pred) > threshold).long()
    else:
        pred_binary = pred
    
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    
    seq_ref = [set(np.where(row)[0]) for row in true]
    seq_pred = [set(np.where(row)[0]) for row in pred_binary]

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
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


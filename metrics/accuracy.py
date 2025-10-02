import torch

def node_level_accuracy(node_logits_batch, labels_batch):
    """
    Compute the accuracy of node classification.

    Parameters:
    - node_logits_batch: Tensor of shape (num_nodes, num_classes), raw output logits from the model.
    - labels_batch: Tensor of shape (num_nodes,), true labels for each node.

    Returns:
    - accuracy: Float, the accuracy of the predictions.
    """

    _, predicted_classes = torch.max(node_logits_batch, dim=1)
    correct_predictions = (predicted_classes == labels_batch).sum().item()
    
    total_predictions = labels_batch.size(0)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy

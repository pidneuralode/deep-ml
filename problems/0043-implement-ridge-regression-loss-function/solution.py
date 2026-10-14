import torch

def ridge_loss(X: torch.Tensor, w: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Implements the Ridge Regression Loss Function using PyTorch.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        y_true: True target values of shape (n_samples,)
        alpha: Regularization parameter (lambda)
    
    Returns:
        The Ridge loss value as a scalar tensor
    """
    # Your implementation here
    y_pred = X @ w
    mse = torch.mean((y_true - y_pred)**2, dim=0)
    l2_penalty = alpha * torch.sum(w ** 2, dim=0)

    return mse + l2_penalty

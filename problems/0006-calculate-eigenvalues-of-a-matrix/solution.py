import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    a, b, c, d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
    delta = torch.sqrt((a+d)**2-4*(a*d-b*c))
    lamda1 = ((a+d)+delta) / 2
    lamda2 = ((a+d)-delta) / 2
    return torch.stack([lamda1, lamda2])

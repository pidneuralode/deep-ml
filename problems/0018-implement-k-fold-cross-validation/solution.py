import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    if k <= 1:
        raise ValueError("k must be at least 2")
        
    # 1. 优雅地生成索引（一行代码搞定打乱或顺序生成）
    indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
    
    # 2. 优雅地切分（array_split 会自动且均匀地处理无法整除的情况）
    folds = np.array_split(indices, k)
    
    # 3. 优雅地组合（使用列表推导式提取当前的测试集和剩余的训练集）
    return [
        (
            np.concatenate(folds[:i] + folds[i+1:]).tolist(),  # Train: 拼接除 i 以外的所有折
            folds[i].tolist()                                  # Test: 第 i 折
        )
        for i in range(k)
    ]

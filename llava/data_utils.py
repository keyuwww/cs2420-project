"""
Shared data utilities for consistent train/test/val splitting
"""

import numpy as np
from typing import List, Dict, Tuple


def split_data(data: List[Dict], train_split: float, val_split: float, 
               test_split: float, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets using consistent method.
    
    This function is used by both evaluate_baseline.py and train_lora.py
    to ensure they use the exact same data splits.
    
    Args:
        data: List of data samples
        train_split: Proportion for training set (default: 0.7)
        val_split: Proportion for validation set (default: 0.15)
        test_split: Proportion for test set (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"
    
    np.random.seed(seed)
    n = len(data)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, val_data, test_data





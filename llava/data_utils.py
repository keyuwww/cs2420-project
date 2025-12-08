
import numpy as np
from typing import List, Dict, Tuple


def split_data(data: List[Dict], train_split: float, val_split: float, 
               test_split: float, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
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





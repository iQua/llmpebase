"""
Useful tools for prompts
"""
from typing import List

import numpy as np


def batch_split(prompts: List[str], batch_num: int):
    """Splitting prompts into batches."""
    n_prompts = len(prompts)
    batch_prompts = np.array_split(prompts, np.ceil(n_prompts / batch_num))
    batch_prompts = [list(batch) for batch in batch_prompts]
    return batch_prompts

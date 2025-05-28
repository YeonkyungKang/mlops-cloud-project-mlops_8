__all__ = ['set_seed', 'SEED']
import os
import random
import numpy as np


# util const
SEED = 1


# 시드 고정
def set_seed(seed: int = SEED) -> None:
    """시드 고정"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"시드 고정 완료: {seed}")
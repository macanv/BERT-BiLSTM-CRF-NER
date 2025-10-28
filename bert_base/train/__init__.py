# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 16:53
 @Author  : MaCan (ma_cancan@163.com)
 @File    : __init__.py.py
"""

from .confidence import aggregate_span_confidence, compute_token_marginals

__all__ = [
    "aggregate_span_confidence",
    "compute_token_marginals",
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集函数优化工具模块
提供离散变量优化的辅助函数
"""

import torch
from torch import Tensor
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.exceptions import InputDataError, UnsupportedError


def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    """将大批量评估分割为多个小批量，避免内存溢出"""
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])


def optimize_acqf_discrete_idx(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 2048,
    unique: bool = True,
) -> Tensor:
    """
    在离散候选集上优化采集函数
    
    对于 q > 1 的情况，使用顺序条件采样（sequential conditioning），
    通过 X_pending 机制考虑已选点对后续选择的影响。
    
    参数:
        acq_function: 采集函数
        q: 需要选择的候选点数量
        choices: 候选点张量，shape (num_choices, d)
        max_batch_size: 最大批量大小，用于分批评估避免内存溢出
        unique: 是否要求返回唯一的候选点（q > 1 时有效）
        
    返回:
        候选点索引张量，shape (q,)
    """
    if isinstance(acq_function, OneShotAcquisitionFunction):
        raise UnsupportedError(
            "离散优化不支持 one-shot 采集函数"
        )
    if choices.numel() == 0:
        raise InputDataError("`choices` 不能为空")
    
    # 为每个候选点添加 batch 维度: (num_choices, 1, d)
    choices_batched = choices.unsqueeze(-2)
    
    if q > 1:
        # 多候选点选择：使用顺序条件采样
        candidate_list = []
        best_idxes = []
        base_X_pending = acq_function.X_pending
        
        # 跟踪原始索引，因为 choices_batched 会被修改
        original_indices = torch.arange(choices.shape[0], device=choices.device)

        # 循环次数不能超过候选点数量
        for _ in range(min(q, choices.shape[0])):
            with torch.no_grad():
                # 批量评估所有候选点的采集值
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
            
            # 选择采集值最高的候选点
            best_idx_in_current = torch.argmax(acq_values)
            original_idx = original_indices[best_idx_in_current]
            best_idxes.append(original_idx)
            
            candidate_list.append(choices_batched[best_idx_in_current])
            
            # 设置 pending points，使采集函数考虑已选点
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            
            # 如果要求唯一性，从候选集中移除已选点
            if unique:
                choices_batched = torch.cat(
                    [choices_batched[:best_idx_in_current], choices_batched[best_idx_in_current + 1 :]]
                )
                original_indices = torch.cat(
                    [original_indices[:best_idx_in_current], original_indices[best_idx_in_current + 1 :]]
                )
        
        # 重置采集函数的 X_pending 状态
        acq_function.set_X_pending(base_X_pending)
        return torch.tensor(best_idxes, dtype=torch.long, device=choices.device)
    
    else:
        # 单候选点选择：直接评估并返回最优索引
        with torch.no_grad():
            acq_values = _split_batch_eval_acqf(
                acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
            )
        best_idx = torch.argmax(acq_values)
        return best_idx.clone().detach()


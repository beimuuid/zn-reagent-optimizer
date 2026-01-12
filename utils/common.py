#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数模块
"""

import re
from pathlib import Path
import pandas as pd
import random
import numpy as np
from typing import Dict, Any, Tuple, Union, Set, FrozenSet, Optional, List
from enum import Enum
import os
import pickle
import glob
import logging
import torch

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保 PyTorch 在 GPU 上的运算是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置 Python 散列种子（虽然在运行时设置效果有限，但对某些库有帮助）
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 如果是较高版本的 torch，可以使用以下设置确保更多算子的确定性
    try:
        torch.use_deterministic_algorithms(True)
        # 某些操作可能需要设置此环境变量以支持确定性
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except (AttributeError, RuntimeError):
        pass

class VarType(Enum):
    """变量类型枚举"""
    CONTINUOUS = "continuous"  # 连续变量，box为(min, max)区间
    DISCRETE = "discrete"  # 离散变量，box为元素集合


def is_continuous(var_name: str, var_types: Optional[Dict[str, str]] = None) -> bool:
    """
    判断变量是否为连续变量
    
    参数:
        var_name: 变量名
        var_types: 变量类型字典，格式为 {"var_name": "continuous" 或 "discrete"}
    """
    if var_types is not None:
        return var_types.get(var_name, "discrete") == "continuous"


def is_discrete(var_name: str, var_types: Optional[Dict[str, str]] = None) -> bool:
    """
    判断变量是否为离散变量
    
    参数:
        var_name: 变量名
        var_types: 变量类型字典，格式为 {"var_name": "continuous" 或 "discrete"}
    """
    if var_types is not None:
        return var_types.get(var_name, "discrete") == "discrete"


def safe_eval_expr(expr: str, vars_dict: Dict[str, Any]) -> float:
    """安全地评估表达式"""
    allowed_builtins = {}
    return eval(expr, {"__builtins__": allowed_builtins}, vars_dict)


def almost_equal_tuple(a, b, tol=1e-6):
    """比较两个元组是否几乎相等"""
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def short(x, n=6):
    """格式化数字为短格式"""
    return float(f"{float(x):.{n}g}")


def _looks_like_evidence_item(d: dict) -> bool:
    """判断字典是否像机理证据条目"""
    if not isinstance(d, dict):
        return False
    keys = set(d.keys())
    return any(k in keys for k in (
        "factor", "label_factor", "response", "mechanism", "label_outcome", "outcome"
    )) and ("name" not in d and "variable" not in d)


_num_pat = re.compile(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)


def _rough_range_from_text(txt: str, fallback: tuple) -> tuple:
    """从文本中粗略提取数值范围"""
    if not isinstance(txt, str):
        return fallback
    nums = [float(m.group(1)) for m in _num_pat.finditer(txt)]
    if len(nums) >= 2:
        lo, hi = sorted([nums[0], nums[1]])
        if hi - lo < 1e-12:
            hi = lo + max(abs(lo) * 0.1, 1e-3)
        return (lo, hi)
    return fallback


def make_json_serializable(obj: Any) -> Any:
    """
    将对象转换为 JSON 可序列化的格式
    """
    if isinstance(obj, set):
        return sorted(list(obj))
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def load_history_csv(path: str, vars: Optional[List[str]] = None, targets: Optional[Dict[str, Any]] = None) -> List[Tuple[Tuple[Any, ...], float]]:
    """
    从CSV加载历史数据（多目标）
    """
    if pd is None:
        raise ImportError("pandas is required to load CSV history. Please install it.")
        
    df = pd.read_csv(path)
    target_cols = list(targets.keys()) if targets else ["yield"]
    for target_col in target_cols:
        if target_col not in df.columns:
            raise ValueError(f"[history] CSV文件缺少目标列: {target_col}")
    
    need_cols = set((vars or []) + target_cols)
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"[history] 缺少数据 {sorted(miss)}，需包含 {sorted(need_cols)}")
        
    df = df.dropna(subset=list(need_cols)).copy()
    rows = []
    for _, r in df.iterrows():
        try:
            var_dict = {v: r[v] for v in (vars or [])}
            target_raw = {target_col: float(r[target_col]) for target_col in target_cols}
            rows.append((var_dict, target_raw))
        except Exception:
            continue
    return rows


def save_checkpoint(state: Dict[str, Any], directory: str, logger: logging.Logger):
    """将优化状态保存到 pickle 文件。"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"checkpoint_round_{state['round']}.pkl")
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"[Checkpoint] 第 {state['round']} 轮的状态已保存至: {filepath}")
    except Exception as e:
        logger.error(f"[Checkpoint] 保存第 {state['round']} 轮的 checkpoint 失败: {e}")


def load_checkpoint(ckpt_path: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """从 pickle 文件加载最新的优化状态。"""
    if not os.path.exists(ckpt_path):
        logger.warning(f"[Checkpoint] Checkpoint 文件 {ckpt_path} 未找到，将从头开始。")
        return None
    
    try:
        with open(ckpt_path, 'rb') as f:
            state = pickle.load(f)
        logger.info(f"[Checkpoint] 成功从 {ckpt_path} 加载 checkpoint。")
        return state
    except Exception as e:
        logger.error(f"[Checkpoint] 从 {ckpt_path} 加载 checkpoint 失败: {e}")
        return None
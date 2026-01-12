from typing import List, Dict, Any
import numpy as np

def aggregate_reward_from_batch(Y_batch: List[Dict[str, float]], 
                                targets: Dict[str, Any],
                                ranges: Dict[str, Any],
                                mode: str = "mean") -> float:
    """
    从一批评估结果中聚合奖励。
    
    该函数处理多目标优化问题，将多个目标值聚合成一个单一的奖励值，
    以便用于MCTS的反向传播。
    
    主要步骤：
    1. 根据 `targets` 将所有目标统一为最大化问题（最小化目标取反）。
    2. 计算所有处理过的目标值的平均值作为最终的单一奖励。
    
    参数:
        Y_batch: 评估结果列表，每个元素是一个包含多个目标值的字典。
                 例如: [{'yield': 95.0, 'cost': 10.0}, {'yield': 92.0, 'cost': 12.0}]
        targets: 目标配置字典，用于指定每个目标是最大化还是最小化。
                       例如: {'yield':'maximize', 'cost':'minimize'}
        ranges: 目标范围字典，用于指定每个目标的范围。
                       例如: {'yield':(0, 100), 'cost':(0, 100)}
        mode: 聚合模式，当前支持 "mean"（平均值）、"max"（最大值）、"min"（最小值）。
        
    返回:
        一个单一的、归一化后的浮点数奖励值。
    """
    if not Y_batch:
        return 0.0
    normalized_value = []
    reward_values = []
    for Y_dict in Y_batch:
        for target_key, raw_value in Y_dict.items():
            if target_key in targets:
                is_maximize = targets.get(target_key) == "maximize"
                if not is_maximize:
                    normalized_value.append(np.clip((float(ranges[target_key][1]) - raw_value) / (float(ranges[target_key][1]) - float(ranges[target_key][0])), 0, 1))
                else:
                    normalized_value.append(np.clip((raw_value - float(ranges[target_key][0])) / (float(ranges[target_key][1]) - float(ranges[target_key][0])), 0, 1))
            #对多目标奖励求平均值(可扩展为加权平均值)
            mean_value = float(np.mean(normalized_value))
            reward_values.append(mean_value)


        # 根据指定的模式聚合所有处理过的奖励值
    if mode == "mean":
        return float(np.mean(reward_values))
    elif mode == "max":
        return float(np.max(reward_values))
    elif mode == "min":
        return float(np.min(reward_values))
    else:
        raise ValueError(f"不支持的聚合模式: {mode}")
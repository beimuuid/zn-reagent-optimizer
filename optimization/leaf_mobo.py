#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
叶内单目标贝叶斯优化（BO）模块
"""

import random
import warnings
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Set, Optional

from utils import is_continuous, is_discrete, set_seed
from agents import PseudoPointPredictionAgent

# 检查 BoTorch 可用性
HAVE_BOTORCH = True
try:
    import torch
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.optim import optimize_acqf
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
    from botorch.models.transforms.outcome import Standardize
    from botorch.sampling.normal import SobolQMCNormalSampler
except Exception as e:
    HAVE_BOTORCH = False
    warnings.warn(f"[WARN] BoTorch 未可用，回退轻量标量化。错误: {e}")


class LeafMOBO:
    """
    叶内贝叶斯优化器（Leaf Bayesian Optimization）
    
    该类实现了在知识树叶节点内的单目标/多目标优化策略，针对产率进行优化，结合了：
    1. 真实观测数据（real observed data）：实际实验获得的数据
    2. 基于 BoTorch 的 qEI 采集函数进行单目标优化
    """
    
    def __init__(self, pseudo_point_prediction_agent: PseudoPointPredictionAgent, batch_q: int = 4, n_restarts: int = 4, raw_samples: int = 8, targets:Dict[str, str] = {"yield":"maximize"},
                 optimization_variables: Dict[str, Any] = None, seed: int = 42):
        """
        初始化叶内单目标贝叶斯优化器
        
        参数:
            batch_q: 每批次推荐的候选点数量，默认为 4
            n_restarts: 多起点优化的重启次数，默认为 4
            raw_samples: 初始随机采样的候选点数量，默认为 8
            targets: 目标列表，如果为None则不使用多目标优化
            vars: 变量列表，如果为None则不使用
            domains: 变量定义域字典，如果为None则不使用
            var_types: 变量类型字典，格式为 {"var_name": "continuous" 或 "discrete"}
            seed: 随机种子，用于保证结果可重现
        """
        self.pseudo_point_prediction_agent = pseudo_point_prediction_agent
        self.q = batch_q  # 批次大小
        self.use_botorch = HAVE_BOTORCH  # 确保 BoTorch 可用
        self.targets = targets if targets is not None else {"yield":"maximize"}  # 目标列表
        self.n_restarts = n_restarts  # 多起点优化的重启次数
        self.raw_samples = raw_samples  # 初始随机采样的候选点数量
        self.seed = seed
        
        # 使用传入的变量和定义域
        self.vars = optimization_variables.get("vars", [])
        self.domains = optimization_variables.get("domains", {})
        self.var_types = optimization_variables.get("var_types", {})
        
        # 真实观测数据存储
        # 使用字典格式支持混合类型变量（连续+离散）
        self.X_real: List[Dict[str, Any]] = []  # 真实输入：变量名到变量值的字典
        self.Y_real: List[Dict[str, float]] = []  # 真实输出：目标值列表（多目标）

        # 伪标签数据存储（基于 LLM 预测生成）
        self.X_pseudo: List[Dict[str, Any]] = []  # 伪标签输入：变量名到变量值的字典
        self.Y_pseudo: List[Dict[str, float]] = []  # 伪标签输出：目标值列表（多目标）
        
        # 最近一次优化的诊断信息
        self.last_diag: Dict[str, Any] = {}
        
        # 变量类型信息（用于编码/解码）
        self.var_names = self.vars  # 变量名列表，按顺序排列
        self.continuous_vars = [v for v in self.var_names if is_continuous(v, self.var_types)]
        self.discrete_vars = [v for v in self.var_names if is_discrete(v, self.var_types)]
        
        # 离散变量的值到索引的映射（用于编码）
        self._discrete_encodings: Dict[str, Dict[Any, int]] = {}
        self._discrete_decodings: Dict[str, List[Any]] = {}
        self._init_discrete_encodings()
        
        # 如果使用 BoTorch，设置 PyTorch 相关配置
        if self.use_botorch:
            self.device = torch.device("cpu")  # 使用 CPU 进行计算
            self.dtype = torch.double  # 使用双精度浮点数以提高数值稳定性

    def _init_discrete_encodings(self):
        """
        初始化离散变量的编码映射
        
        为每个离散变量创建值到索引的映射，用于在优化时将离散值编码为连续索引。
        """
        for var_name in self.discrete_vars:
            domain = self.domains.get(var_name)
            if domain is None:
                continue
            if isinstance(domain, set):
                values = sorted(list(domain))  # 排序以保证一致性
            else:
                values = list(domain) if hasattr(domain, '__iter__') else [domain]
            
            # 创建值到索引的映射（0, 1, 2, ...）
            encoding = {val: idx for idx, val in enumerate(values)}
            decoding = values  # 索引到值的映射
            
            self._discrete_encodings[var_name] = encoding
            self._discrete_decodings[var_name] = decoding

    def _encode_var_dict(self, var_dict: Dict[str, Any]) -> np.ndarray:
        """
        将变量字典编码为数值向量
        
        连续变量：直接使用其值（归一化后）
        离散变量：使用整数索引（0, 1, 2, ...）
        
        参数:
            var_dict: 变量名到变量值的字典
            
        返回:
            np.ndarray: 编码后的数值向量，shape (n_vars,)
        """
        encoded = []
        for var_name in self.var_names:
            if var_name in var_dict:
                value = var_dict[var_name]
                if is_continuous(var_name, self.var_types):
                    # 连续变量：直接使用浮点值
                    encoded.append(float(value))
                else:
                    # 离散变量：使用索引
                    encoding = self._discrete_encodings.get(var_name, {})
                    idx = encoding.get(value, 0)
                    encoded.append(float(idx))
            else:
                # 缺失变量：使用默认值
                encoded.append(0.0)
        return np.array(encoded, dtype=float)

    def _decode_var_array(self, encoded_array: np.ndarray, 
                          leaf_box: Dict[str, Union[Tuple[float, float], Set[Any]]]) -> Dict[str, Any]:
        """
        将编码后的数值向量解码为变量字典
        
        连续变量：从归一化空间反变换回原始值，并裁剪到 leaf_box
        离散变量：从索引映射回原始值，并确保选择 leaf_box 中最近的有效值
        
        参数:
            encoded_array: 编码后的数值向量，shape (n_vars,)
            leaf_box: 叶节点的参数边界（用于连续变量的反归一化和离散变量的约束）
            
        返回:
            Dict[str, Any]: 变量名到变量值的字典
        """
        var_dict = {}
        for idx, var_name in enumerate(self.var_names):
            if idx >= len(encoded_array):
                continue
                
            if is_continuous(var_name, self.var_types):
                # 连续变量：从归一化值反变换
                # 使用全局定义域进行反归一化
                global_domain = self.domains.get(var_name)
                g_lo, g_hi = global_domain
                # encoded_array[idx] 是归一化后的值 [0, 1]
                # 反变换回全局空间
                value = encoded_array[idx] * (g_hi - g_lo) + g_lo
                # 如果 leaf_box 有约束，则裁剪到叶节点范围
                if var_name in leaf_box:
                    lo, hi = leaf_box[var_name]
                    value = np.clip(value, lo, hi)
                else:
                    value = np.clip(value, g_lo, g_hi)
                var_dict[var_name] = float(value)
            else:
                # 离散变量：从索引映射回值
                decoding = self._discrete_decodings.get(var_name, [])
                if decoding:
                    # 反归一化到索引空间 [0, len-1]
                    max_idx = max(1, len(decoding) - 1)
                    idx_float = encoded_array[idx] * max_idx
                    
                    # 确定允许的索引集合
                    allowed_indices = []
                    if var_name in leaf_box:
                        box_val = leaf_box[var_name]
                        # box_val 是值的集合，需转为索引
                        encoding = self._discrete_encodings.get(var_name, {})
                        allowed_indices = [encoding[v] for v in box_val if v in encoding]
                        allowed_indices.sort()
                    
                    if not allowed_indices:
                        # 如果没有找到允许的索引，或者 leaf_box 中没有该变量，默认所有索引
                        allowed_indices = list(range(len(decoding)))
                    
                    # 找到最近的索引
                    # idx_float 是浮点数，allowed_indices 是整数列表
                    # 使用 numpy 计算最小距离
                    allowed_arr = np.array(allowed_indices)
                    if len(allowed_arr) > 0:
                        closest_idx_in_allowed = np.abs(allowed_arr - idx_float).argmin()
                        final_idx = allowed_indices[closest_idx_in_allowed]
                        var_dict[var_name] = decoding[final_idx]
        
        return var_dict

    def feasible_dict(self, var_dict: Dict[str, Any]) -> bool:
        """
        检查一组参数是否可行,目前只做简单检查
        
        参数:
            var_dict: 变量名到变量值的字典，支持连续和离散变量
            
        返回:
            bool: 如果满足所有约束则返回 True，否则返回 False
        """
        for var_name, value in var_dict.items():
            if var_name in self.domains:
                domain = self.domains.get(var_name)
                if is_continuous(var_name, self.var_types):
                    if not (domain[0] <= value <= domain[1]):
                        return False
                else:
                    if value not in domain:
                        return False
        return True
    def _seed_pseudo_with_llm(self,
                             box: Dict[str, Union[Tuple[float, float], Set[Any]]], 
                             reactants: Dict[str, Any] = {},
                             mechanism_structured_data: Optional[List[Dict[str, Any]]] = None,
                             n_samples_per_continuous_var: int = 10):
        """
        使用 LLM 基于文献知识预测产率生成伪标签（内部实现）
        
        对于离散变量：枚举所有可能的组合
        对于连续变量：在范围内均匀采样（每个变量采样n_samples_per_continuous_var个点）
        生成所有可能的组合，不受数量限制
        
        参数:
            box: 参数边界字典
                - 连续变量: (下界, 上界) 的元组
                - 离散变量: 可选值的集合
            reactants: 反应物信息字典
            mechanism_structured_data: 文献知识结构化数据（如果使用文献知识）
            variables_extractor: VariablesExtractor 实例
            n_samples_per_continuous_var: 每个连续变量采样数
        """
        from itertools import product
        
        # 分离离散变量和连续变量
        discrete_vars_in_box = [v for v in self.discrete_vars if v in box]
        continuous_vars_in_box = [v for v in self.continuous_vars if v in box]
        
        # 为离散变量生成所有可能的值组合
        discrete_combinations = []
        if discrete_vars_in_box:
            discrete_domains = []
            for var_name in discrete_vars_in_box:
                domain = box[var_name]
                if isinstance(domain, set):
                    values = sorted(list(domain))
                elif isinstance(domain, (list, tuple)):
                    values = sorted(list(domain))
                else:
                    values = [domain]
                discrete_domains.append([(var_name, val) for val in values])
            
            # 生成所有离散变量的组合
            for combo in product(*discrete_domains):
                discrete_dict = {var_name: val for var_name, val in combo}
                discrete_combinations.append(discrete_dict)
        else:
            discrete_combinations = [{}]
        
        # 为连续变量生成均匀采样值
        # 对每个连续变量在范围内均匀采样
        if len(continuous_vars_in_box) > 0:
            # 每个连续变量采样10个点（可根据需要调整）
            n_samples_per_var = n_samples_per_continuous_var
        else:
            n_samples_per_var = 1  # 默认值，实际上不会使用
        
        continuous_samples = []
        if continuous_vars_in_box:
            for var_name in continuous_vars_in_box:
                lo, hi = box[var_name]
                # 在范围内均匀采样
                samples = np.linspace(lo, hi, n_samples_per_var)
                continuous_samples.append([(var_name, val) for val in samples])
        else:
            continuous_samples = [[]]
        
        # 生成所有候选点（离散组合 × 连续采样）
        var_dicts = []
        for discrete_dict in discrete_combinations:
            if continuous_vars_in_box:
                for continuous_combo in product(*continuous_samples):
                    var_dict = dict(discrete_dict)
                    var_dict.update({var_name: val for var_name, val in continuous_combo})
                    # 检查可行性约束
                    if self.feasible_dict(var_dict):
                        var_dicts.append(var_dict)
            else:
                if self.feasible_dict(discrete_dict):
                    var_dicts.append(discrete_dict)
        
        if not var_dicts:
            print(f"[LeafMOBO] 警告：无法生成有效的反应条件")
            return
        
        # 构建反应信息字典（合并反应物和反应条件）
        reaction_infos = []
        for var_dict in var_dicts:
            if reactants is not {} and len(reactants) > 0:
                reaction_info = dict(reactants)  # 复制反应物信息
            else:
                reaction_info = {}
            # 添加反应条件（ligand, base, solvent 等）
            for var_name in self.var_names:
                if var_name in var_dict:
                    reaction_info[var_name] = var_dict[var_name]
            reaction_infos.append(reaction_info)

        # 调用 LLM 批量预测
        try:
            predictions = self.pseudo_point_prediction_agent.predict_pseudo_points_from_knowledge(
                    mechanism_structured_data=mechanism_structured_data,
                    reaction_info=reaction_infos,
                    targets=self.targets
                )
            
            # 处理预测结果
            # 确保 predictions 和 var_dicts 数量一致
            min_len = min(len(predictions), len(var_dicts))
            if min_len < len(var_dicts):
                print(f"[LeafMOBO] 警告：预测结果数量({len(predictions)})少于反应条件数量({len(var_dicts)})")
            
            for i in range(min_len):
                var_dict = var_dicts[i]
                pred_result = predictions[i]
                # 存储伪标签数据
                self.X_pseudo.append(var_dict)
                self.Y_pseudo.append(pred_result.get('predicted_targets', {}))
            
            print(f"[LeafMOBO] 使用 LLM 成功生成 {len(self.X_pseudo)} 个伪标签样本")
            
        except Exception as e:
            print(f"[LeafMOBO] LLM 预测失败：{e}")
            raise RuntimeError(f"LLM 预测失败，无法生成伪标签：{e}") from e
    
    def observe_real(self, X: List[Dict[str, Any]], Y: List[Dict[str, float]]):
        """
        添加真实实验观测数据
        
        将新的实验数据加入到真实数据集中，用于更新高斯过程模型。
        
        参数:
            X: 实验输入参数列表，每个元素为变量名到变量值的字典
            Y: 对应的实验输出（目标值）列表，每个元素为目标名到目标值的字典
        """
        self.X_real.extend(X)
        self.Y_real.extend(Y)

    def prune_pseudo(self, sim_th: float = 0.98, drop_ratio: float = 0.2):
        """
        剪枝伪标签数据：移除与真实数据过于相似的伪数据
        
        当获得真实实验数据后，与真实数据过于接近的伪标签可能会引入偏差或冗余。
        该方法通过余弦相似度筛选伪数据，保留与真实数据有一定距离的样本。
        
        策略:
        1. 计算每个伪数据点与所有真实数据点的余弦相似度
        2. 移除相似度超过阈值的伪数据
        3. 对保留的数据随机丢弃一定比例，避免过拟合伪标签
        
        参数:
            sim_th: 余弦相似度阈值，默认 0.98（高于此值则认为过于相似）
            drop_ratio: 额外随机丢弃的比例，默认 0.2（丢弃 20%）
        """
        # 如果没有真实数据或伪数据，则无需剪枝
        if not self.X_real or not self.X_pseudo:
            return

        def feat(x_dict):
            """
            将输入参数归一化到 [0, 1] 范围
            
            归一化消除不同维度的量纲影响，使余弦相似度计算更准确。
            支持连续和离散变量。
            """
            features = []
            for var_name in self.var_names:
                if var_name not in x_dict:
                    features.append(0.0)
                    continue
                    
                value = x_dict[var_name]
                if is_continuous(var_name, self.var_types):
                    # 连续变量：Min-Max 归一化
                    domain = self.domains.get(var_name)
                    if isinstance(domain, tuple) and len(domain) == 2:
                        lo, hi = domain
                        norm_val = (value - lo) / (hi - lo + 1e-9)
                        features.append(float(np.clip(norm_val, 0.0, 1.0)))
                    else:
                        features.append(0.0)
                else:
                    # 离散变量：使用索引归一化
                    encoding = self._discrete_encodings.get(var_name, {})
                    if value in encoding:
                        idx = encoding[value]
                        decoding = self._discrete_decodings.get(var_name, [])
                        if decoding:
                            # 归一化到 [0, 1]
                            norm_val = idx / max(1, len(decoding) - 1)
                            features.append(float(norm_val))
                        else:
                            features.append(0.0)
                    else:
                        features.append(0.0)
            
            return np.array(features, dtype=float)

        # 将所有真实数据转换为归一化特征矩阵
        R = np.stack([feat(x) for x in self.X_real], axis=0)  # shape: (n_real, n_vars)
        
        keep = []  # 保留的伪数据索引列表
        for i, x in enumerate(self.X_pseudo):
            v = feat(x)  # 归一化伪数据点
            # 计算与所有真实数据的余弦相似度，取最大值
            cos = np.max(np.dot(R, v) / (np.linalg.norm(R, axis=1) * np.linalg.norm(v) + 1e-9))
            # 只保留相似度低于阈值的伪数据
            if cos < sim_th:
                keep.append(i)
        
        # 如果所有伪数据都被过滤掉，至少保留一半
        if not keep:
            keep = list(range(min(len(self.X_pseudo) // 2 + 1, len(self.X_pseudo))))
        
        # 随机打乱并丢弃一部分数据，增加多样性，避免过度依赖伪标签
        random.shuffle(keep)
        cut = int(len(keep) * (1 - drop_ratio))
        keep = keep[:max(1, cut)]  # 至少保留 1 个
        
        # 更新伪数据集
        self.X_pseudo = [self.X_pseudo[i] for i in keep]
        self.Y_pseudo = [self.Y_pseudo[i] for i in keep]
    

    def _suggest_scalarized(self, leaf_box: Dict[str, Union[Tuple[float, float], Set[Any]]], k: int) -> List[Dict[str, Any]]:
        """
        回退方案：使用随机采样和随机分数来推荐候选点
        
        当 BoTorch 不可用或数据量不足时，使用这个轻量级的替代方法。
        
        算法步骤:
        1. 在叶节点空间内随机采样大量候选点（默认1200个）。
        2. 检查可行性约束。
        3. 使用随机分数。
        4. 选择分数最高的 k 个点作为推荐。
        
        参数:
            leaf_box: 叶节点的参数边界
                - 连续变量: (下界, 上界) 的元组
                - 离散变量: 可选值的集合
            k: 需要推荐的候选点数量
            
        返回:
            推荐的候选点列表，每个元素为变量字典
        """
        # 固定随机种子
        
        pool = []  # 候选池：存储 (分数, 参数字典) 对
        num_samples = k * 100
        # 生成大量随机候选点
        for _ in range(num_samples):
            var_dict = {}  # 构建变量字典
            
            # 为每个变量采样值
            for var_name in self.var_names:
                if var_name not in leaf_box:
                    continue
                    
                if is_continuous(var_name, self.var_types):
                    # 连续变量：在范围内均匀随机采样
                    lo, hi = leaf_box[var_name]
                    var_dict[var_name] = random.uniform(lo, hi)
                else:
                    # 离散变量：从允许的值集合中随机选择
                    domain = leaf_box[var_name]
                    values = sorted(list(domain))
                    if values:
                        var_dict[var_name] = random.choice(values)
                        
            
            # 检查可行性
            if not self.feasible_dict(var_dict):
                continue
            
            # 使用随机得分
            score = random.random()
            pool.append((score, var_dict))

        
        # 按聚合奖励降序排序，选择最优的候选点
        pool.sort(key=lambda x: x[0], reverse=True)
        
        # 记录诊断信息
        self.last_diag = {
            "backend": "random_sampling",
            "q": k,
            "note": "BoTorch disabled or data too few; used random sampling."
        }
        
        # 返回前 k 个最优候选点
        return [p[1] for p in pool[:k]]

    def _to_torch(self, X: List[Dict[str, Any]], Y: List[Dict[str, float]]):
        """
        将数据转换为 PyTorch 张量,并将输入X归一化到[0,1]

        参数:
            X: 输入参数列表（变量字典）
            Y: 输出目标值列表（目标值字典列表）

        返回:
            X_t: 输入张量，shape: (n_samples, n_vars)
            Y_t: 输出张量，shape: (n_samples, n_targets)
        """
        X_encoded = []
        for x_dict in X:
            encoded = []
            for var_name in self.var_names:
                if var_name not in x_dict:
                    encoded.append(0.0)
                    continue
                    
                value = x_dict[var_name]
                if is_continuous(var_name, self.var_types):
                    # 连续变量：Min-Max 归一化
                    domain = self.domains.get(var_name)
                    lo, hi = domain
                    norm_val = (value - lo) / (hi - lo + 1e-9)
                    encoded.append(float(np.clip(norm_val, 0.0, 1.0)))
                else:
                    # 离散变量：使用索引归一化
                    encoding = self._discrete_encodings.get(var_name, {})
                    if value in encoding:
                        idx = encoding[value]
                        decoding = self._discrete_decodings.get(var_name, [])
                        if decoding:
                            # 归一化到 [0, 1]
                            norm_val = idx / max(1, len(decoding) - 1)
                            encoded.append(float(norm_val))
                        else:
                            encoded.append(0.0)
                    else:
                        encoded.append(0.0)
            X_encoded.append(encoded)
        
        X_np = np.array(X_encoded, dtype=float)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        Y_np = np.array([[y[target] for target in self.targets.keys()] for y in Y], dtype=float).reshape(-1, len(self.targets.keys()))
        X_t = torch.tensor(X_np, dtype=self.dtype, device=self.device)
        Y_t = torch.tensor(Y_np, dtype=self.dtype, device=self.device)
        return X_t, Y_t

    def _point_in_leaf_box(self, var_dict: Dict[str, Any], leaf_box: Dict[str, Union[Tuple[float, float], Set[Any]]]) -> bool:
        """
        检查一个参数字典是否属于给定的叶子节点box
        
        参数:
            var_dict: 变量名到变量值的字典
            leaf_box: 叶节点的参数边界字典
            
        返回:
            bool: 如果所有变量都在leaf_box范围内则返回True
        """
        for var_name in self.var_names:
            if var_name not in var_dict:
                continue
            if var_name not in leaf_box:
                # 如果leaf_box中没有该变量，检查是否在全局定义域内
                continue
            
            value = var_dict[var_name]
            box_value = leaf_box[var_name]
            
            if is_continuous(var_name, self.var_types):
                lo, hi = box_value
                if not (lo <= value <= hi):
                    return False
            else:
                if value not in box_value:
                    return False
        
        return True

    def suggest(self, leaf_box: Dict[str, Union[Tuple[float, float], Set[Any]]], k: int = 5) -> List[Dict[str, Any]]:
        """
        推荐下一批候选实验点（核心方法）
        
        使用基于 BoTorch 的贝叶斯优化算法（qEI）推荐下一批最有潜力的候选点。
        该方法针对目标值进行优化，使用所有真实数据训练高斯过程，
        并在目标叶子节点空间内进行连续优化，对离散变量进行四舍五入和最近邻匹配。
        
        算法流程:
        1. 使用所有真实数据训练高斯过程（GP）模型
        2. 计算叶子节点在归一化空间中的边界
        3. 优化采集函数，找到最优的一批连续候选点
        4. 解码候选点，将离散变量还原为 leaf_box 中最近的有效值
        
        参数:
            leaf_box: 叶节点的参数边界字典
                - 连续变量: (下界, 上界) 的元组
                - 离散变量: 可选值的集合
            k: 需要推荐的候选点数量（可能小于等于 self.q）
            
        返回:
            推荐的候选点列表，每个元素为变量字典
        """
        # 固定随机种子

        # 如果 BoTorch 不可用，使用回退方案
        if not self.use_botorch:
            print("BoTorch 不可用，回退到标量化方法")
            return self._suggest_scalarized(leaf_box, k)
        
        if len(self.X_pseudo) > 0 and len(self.Y_pseudo) > 0:
            # 合并伪标签数据和真实数据，并过滤出属于当前叶子节点的数据
            # 只使用属于当前leaf_box的数据来训练模型，避免不同叶子节点的数据混合
            X_all = self.X_pseudo + self.X_real
            Y_all = self.Y_pseudo + self.Y_real
            
            # 过滤出属于当前叶子节点的数据
            X = []
            Y = []
            for x_dict, y_val in zip(X_all, Y_all):
                if self._point_in_leaf_box(x_dict, leaf_box):
                    X.append(x_dict)
                    Y.append(y_val)
        else:
            # 只使用真实数据
            X = self.X_real
            Y = self.Y_real
        
        # 如果数据量不足，回退到标量化方法
        # 高斯过程需要足够的数据才能有效建模
        if len(X) < 4:
            print(f"数据量不足，回退到随机采样: {len(X)} < 4")
            return self._suggest_scalarized(leaf_box, k)

        # 转换为 PyTorch 张量（已经归一化）
        Xt, Yt = self._to_torch(X, Y)
        
        # 计算 leaf_box 在归一化空间中的边界 (2, n_vars)
        bounds_list = []
        for var_name in self.var_names:
            # 默认为 [0, 1]
            b_lo, b_hi = 0.0, 1.0
            
            if var_name in leaf_box:
                # 限制在 leaf_box 范围内
                if is_continuous(var_name, self.var_types):
                    # 连续变量
                    g_domain = self.domains.get(var_name)
                    l_domain = leaf_box[var_name]
                    g_lo, g_hi = g_domain
                    l_lo, l_hi = l_domain
                    denom = g_hi - g_lo + 1e-9
                    b_lo = (l_lo - g_lo) / denom
                    b_hi = (l_hi - g_lo) / denom
                else:
                    # 离散变量
                    # 找到 leaf_box 中允许的最小和最大索引
                    decoding = self._discrete_decodings.get(var_name, [])
                    encoding = self._discrete_encodings.get(var_name, {})
                    l_vals = leaf_box[var_name]
                    indices = [encoding[v] for v in l_vals if v in encoding]                   
                    if indices and decoding:
                        min_idx = min(indices)
                        max_idx = max(indices)
                        denom = max(1, len(decoding) - 1)
                        b_lo = min_idx / denom
                        b_hi = max_idx / denom
            
            # Clip to [0, 1] just in case
            b_lo = max(0.0, min(1.0, b_lo))
            b_hi = max(0.0, min(1.0, b_hi))
            bounds_list.append([b_lo, b_hi])
        
        bounds = torch.tensor(bounds_list, dtype=self.dtype, device=self.device).t()  # (2, n_vars)
        
        # 输入数据已经在 _to_torch 中归一化，直接使用
        Xn = Xt

        # Yt 已经是 (n_samples, n_targets) 的形状，表示目标值
        y = Yt  # 目标值
        
        if len(self.targets.keys()) == 1:
            # 单目标: qEI
            model = SingleTaskGP(Xn, y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            best_f = y[:, 0].max().item()
            acq = qLogExpectedImprovement(
                model=model,
                best_f=best_f,
            )
        else:
            # 多目标: 为每个目标拟合独立的高斯过程
            weights = []
            for key, direction in self.targets.items():
                # 对于最小化目标，将其转化为最大化问题
                if direction == "maximize":
                    weights.append(1.0)
                else:  # 'minimize'
                    weights.append(-1.0)
            weights = torch.tensor(weights, dtype=self.dtype, device=self.device)
            
            # 转换目标，使得所有目标都是最大化
            y_transformed = y * weights
            
            # 为每个目标拟合一个独立的 GP 模型
            models = []
            for i in range(y_transformed.shape[1]):
                model_i = SingleTaskGP(
                    Xn, y_transformed[:, [i]], outcome_transform=Standardize(m=1)
                )
                mll_i = ExactMarginalLogLikelihood(model_i.likelihood, model_i)
                fit_gpytorch_mll(mll_i)
                models.append(model_i)
            
            # 使用 ModelListGP 将多个模型组合
            model = ModelListGP(*models)
            
            # 定义超体积计算的参考点 (reference point)
            # 一个好的实践是使用一个被所有观测值支配的点
            ref_point = y_transformed.min(0).values
            
            # 使用 qNEHVI (q-Noisy Expected Hypervolume Improvement) 作为采集函数
            acq = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                X_baseline=Xn,
                prune_baseline=True,
            )

        # 记录诊断信息
        diag_info = {
            "q": self.q,  # 批次大小
            "baseline_n": int(Xn.shape[0]),  # 基线数据点数量
            "outcome_transform": "Standardize(m=1)",  # 输出变换类型
            "bounds": bounds.detach().cpu().numpy().tolist()
        }
        if len(self.targets.keys()) == 1:
            diag_info["backend"] = "continuous_qEI"
            diag_info["best_f"] = float(best_f)
        else:
            diag_info["backend"] = "qNEHVI_continuous"
            diag_info["ref_point"] = ref_point.detach().cpu().numpy().tolist()
        self.last_diag = diag_info

        # 使用连续优化方法优化采集函数
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=self.q,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        
        # 将归一化的连续候选点解码为原始空间
        out = []
        candidates_np = candidates.detach().cpu().numpy()

        for i in range(candidates_np.shape[0]):
            cand_vector = candidates_np[i]
            # 解码并还原离散变量

            var_dict = self._decode_var_array(cand_vector, leaf_box)
            out.append(var_dict)
        
        # 返回前 k 个候选点
        if len(out) < k:
            return out
        return out[:k]

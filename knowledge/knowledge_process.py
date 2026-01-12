#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识融合模块
整合机理知识和文献先验，为优化提供融合的先验信息
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Set, Union, FrozenSet, Optional

from utils import is_continuous, is_discrete


class KnowledgeProcessor:
    """
    文献知识处理类
    
    该类将大模型提取到的变量划分处理为后续知识树构建所需的结构，
    生成统一的重要性评分、区间切分和聚类。
    """
    
    def __init__(self, variable_divisions: Dict[str, Any], 
                 optimization_variables: Optional[Dict[str, Any]] = None):
        """
        初始化知识处理器
        
        参数:
            variable_divisions (Dict[str, Any]): 变量划分
                                 格式: {"variable_divisions": [...], "importance": {...}}
            optimization_variables: 优化变量，格式为 {"vars": [...], "domains": {...}, "var_types": {...}}
        """
        # 使用传入的变量和定义域
        self.vars = optimization_variables.get("vars", [])
        self.domains = optimization_variables.get("domains", {})
        self.var_types = optimization_variables.get("var_types", {})
        # 从变量划分中提取重要性
        self.variable_divisions = variable_divisions.get("variable_divisions", [])
        self.importance = variable_divisions.get("importance", {})
        
        # 为每个变量提取切分信息
        self.splits = {}  # 连续变量的区间切分
        self.clusters = {}  # 离散变量的聚类

        for v in self.vars: # 对每个变量进行处理
            if is_continuous(v, self.var_types): # 连续变量
                self.splits[v] = self._merge_splits(v) # 提取区间切分
            else: # 离散变量
                self.clusters[v] = self._merge_clusters(v) # 提取聚类

    def _merge_splits(self, v: str) -> List[Tuple[float, float]]:
        """
        提取指定变量的切分区间
        
        将文献中提到的范围边界提取，生成统一的区间列表。
        
        参数:
            v (str): 变量名
            
        返回:
            List[Tuple[float, float]]: 区间列表，每个元素为 (下界, 上界) 的元组
        """
        # 获取该变量的定义域范围
        domain = self.domains.get(v)
        if domain is None:
            return []
        if len(domain) == 2:
            dom_lo, dom_hi = domain
        else:
            return []  # 离散变量不需要切分
        # 从机理配置中获取已有的切分点
        cuts = set()
        
        # 遍历文献事实，提取与该变量相关的范围边界
        for f in self.variable_divisions:
            # 情况1：文献事实直接指定了该变量
            if f.get("variable") == v and "range" in f:
                lo, hi = f["range"]
                # 将范围的下界添加为切分点（需在定义域内）
                if lo is not None and dom_lo <= lo <= dom_hi:
                    cuts.add(float(lo))
                # 将范围的上界添加为切分点（需在定义域内）
                if hi is not None and dom_lo <= hi <= dom_hi:
                    cuts.add(float(hi))
        
        # 筛选切分点：只保留严格在定义域内部的点（不包括边界）
        cuts = [x for x in sorted(list(cuts)) if dom_lo < x < dom_hi]
        # 构建边界列表：定义域下界 + 内部切分点 + 定义域上界
        edges = [dom_lo] + cuts + [dom_hi]
        # 返回区间列表：将相邻边界两两配对形成区间
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def _merge_clusters(self, v: str) -> List[Set[Any]]:
        """
        提取指定离散变量的聚类
        
        将文献中提到的聚类提取，生成统一的聚类列表。
        
        参数:
            v (str): 离散变量名
            
        返回:
            List[Set[Any]]: 聚类列表，每个聚类是一个元素集合
        """
        # 获取该变量的定义域（所有可能的取值）
        domain_raw = self.domains.get(v)
        if domain_raw is None:
            return []
        if isinstance(domain_raw, set):
            domain = domain_raw
        elif isinstance(domain_raw, (list, tuple)):
            domain = set(domain_raw)
        else:
            domain = set([domain_raw])
        if not isinstance(domain, set):
            domain = set(domain) if hasattr(domain, '__iter__') else set()
        
        all_clusters = []
        
        # 从文献事实中提取相关的离散值集合
        lit_suggested_sets = []
        for f in self.variable_divisions:
            # 情况1：文献事实直接指定了该变量的取值集合
            if f.get("variable") == v and "cluster" in f:
                cluster = f["cluster"]
                if isinstance(cluster, (list)):
                    # 只保留在定义域内的值
                    valid_cluster = set(cluster) & domain
                    if valid_cluster:
                        lit_suggested_sets.append(valid_cluster)
        
        # 将文献建议的集合加入聚类候选
        all_clusters.extend(lit_suggested_sets)
        
        # 如果没有任何聚类配置，返回默认：整个定义域作为一个聚类
        if not all_clusters:
            return [domain] if domain else []
        
        # 合并重叠的聚类（使用并查集算法）
        # 确保每个元素只属于一个聚类
        merged_clusters = self._merge_overlapping_clusters(all_clusters, domain)
        
        return merged_clusters
    
    def _merge_overlapping_clusters(self, clusters: List[Set[Any]], domain: Set[Any]) -> List[Set[Any]]:
        """
        合并有交集的文献聚类
        
        使用并查集思想，将有交集的文献聚类合并为一个聚类。
        最后将未被任何聚类覆盖的元素单独作为聚类。
        
        参数:
            clusters: 初始文献聚类列表（可能有重叠）
            domain: 变量的完整定义域
            
        返回:
            合并后的文献聚类列表（无重叠）
        """
        if not clusters:
            return [domain] if domain else []
        
        # 使用并查集合并有交集的聚类
        merged = []
        for cluster in clusters:
            if not cluster:
                continue
            # 检查当前聚类是否与已有聚类有交集
            found = False
            for i, existing in enumerate(merged):
                if cluster & existing:  # 有交集，合并
                    merged[i] = existing | cluster
                    found = True
                    break
            if not found:
                # 没有交集，作为新聚类
                merged.append(set(cluster))
        
        # 再次合并（处理传递性：A∩B≠∅, B∩C≠∅ => A∪B∪C）
        # 重复合并直到没有可合并的聚类
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = set()
            for i, c1 in enumerate(merged):
                if i in used:
                    continue
                combined = set(c1)
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    if combined & merged[j]:  # 有交集
                        combined |= merged[j]
                        used.add(j)
                        changed = True
                used.add(i)
                new_merged.append(combined)
            merged = new_merged
        
        # 收集所有已被聚类覆盖的元素
        covered = set()
        for c in merged:
            covered |= c
        
        # 将未被覆盖的元素单独作为聚类（如果有）
        uncovered = domain - covered
        if uncovered:
            # 可以选择：每个未覆盖元素单独成为一个聚类
            # 或者：所有未覆盖元素合并为一个"其他"聚类
            # 这里采用后者
            merged.append(uncovered)
        
        # 过滤空聚类
        return [c for c in merged if c]

    def prior_score_variable_divisions(self, box: Dict[str, Union[Tuple[float, float], Set[Any]]]) -> float:
        """
        计算基于变量划分的先验得分
        
        对给定的超矩形/超集合区域（box），计算基于变量划分的先验得分。
        得分反映了该区域包含高性能实验点的可能性。
        
        参数:
            box: 参数空间区域，键为变量名，值为：
                 - 连续变量：(下界, 上界) 元组
                 - 离散变量：包含取值的集合
            
        返回:
            float: 基于变量划分的先验得分，范围 [0, 1]，值越高表示该区域越有前景
        """
        # 用于存储变量划分对该区域的贡献
        contrib = []
        
        # 遍历所有变量划分，计算每个变量划分对该区域的贡献
        for f in self.variable_divisions:
            var_name = f.get("variable")
            # 情况1：文献事实针对单个连续变量的范围
            if var_name in box.keys() and is_continuous(var_name, self.var_types) and "range" in f:
                box_val = box[var_name]
                # 获取该变量在box中的范围
                lo, hi = box_val[0], box_val[1]
                # 获取文献中推荐的范围
                rlo, rhi = f.get("range", [None, None])
                if rlo is not None and rhi is not None:
                    # 计算box与文献推荐范围的交集长度
                    inter = max(0.0, min(hi, rhi) - max(lo, rlo))
                    # 计算box的范围长度（防止除零）
                    length = max(1e-9, hi - lo)
                    # 覆盖率：交集占box的比例
                    cover = inter / length
                    # 获取变量划分的置信度，默认0.5
                    conf = float(f.get("confidence", 0.5))
                    # 获取文献事实的符号：+ 正面，0 中性，- 负面
                    sign = f.get("sign", "+")
                    # 计算增益：覆盖率 * 置信度 * 符号权重
                    # 正面(+):1.0, 中性(0):0.5, 负面(-):0.2
                    gain = cover * conf * (1.0 if sign == "+" else 0.5 if sign == "0" else 0.2)
                    if gain > 0:
                        contrib.append(gain)            
            # 情况1b：变量划分针对单个离散变量的取值集合
            elif var_name in box.keys() and is_discrete(var_name, self.var_types) and "cluster" in f:
                box_val = box[var_name]
                # 获取变量划分中推荐的取值集合
                variable_divisions_values = f.get("cluster", [])
                if isinstance(variable_divisions_values, (list, set)):
                    variable_divisions_set = set(variable_divisions_values)
                    # 计算交集
                    intersection = box_val & variable_divisions_set
                    # 覆盖率：交集占box的比例
                    cover = len(intersection) / max(1, len(box_val))
                    conf = float(f.get("confidence", 0.5))
                    sign = f.get("sign", "+")
                    gain = cover * conf * (1.0 if sign == "+" else 0.5 if sign == "0" else 0.2)
                    if gain > 0:
                        contrib.append(gain)            
        
        # 计算变量划分的先验得分：所有贡献的平均值，限制在[0,1]范围内
        s_variable_divisions = float(np.clip(np.mean(contrib), 0, 1)) if contrib else 0.0
        return float(np.clip(s_variable_divisions, 0, 1))


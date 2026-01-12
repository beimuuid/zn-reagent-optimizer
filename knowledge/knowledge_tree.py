#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识树模块（带UCB选择）
实现基于化学先验知识的分层搜索空间，使用UCB策略进行探索-利用平衡
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import random # Added for random sampling in select_leaf_by_ucb

from utils import is_continuous, is_discrete
from .knowledge_process import KnowledgeProcessor


@dataclass
class Node:
    node_id: int
    depth: int
    box: Dict[str, Tuple[float,float]]
    var_name: Optional[str] = None
    parent_id: Optional[int] = None
    children: List["Node"] = field(default_factory=list)
    n: float = 0.0
    Q: float = 0.0
    prior_n: float = 0.0
    prior_Q: float = 0.0
    decay_rate: float = 3.0

    def mean(self) -> float:
        # 改进：先验衰减机制 - 随真实访问次数增加，先验权重衰减
        n_tot = self.n + self.prior_n  # 总有效访问次数 = 实际 + 先验
        return 0.0 if n_tot <= 0 else (self.Q + self.prior_Q) / n_tot  # 平均奖励 = 总奖励 / 总次数

    def expl(self, Nparent: float, Cp: float=2.0, alpha: float=1.0) -> float:
        n_eff = self.n + self.prior_n
        # 避免log(1)导致探索项过大
        log_term =  math.log(max(1, Nparent))
        expl = Cp * math.sqrt(max(0.0, log_term / (n_eff + alpha)))
        return expl

    def ucb(self, Nparent: float, Cp: float=2.0, alpha: float=1.0) -> float:
        return self.mean() + self.expl(Nparent, Cp, alpha)


class KnowledgeTree:
    """
    知识树类：基于融合知识构建的分层搜索空间
    
    实现了一个层次化的多臂赌博机树，结合了：
    1. 化学先验知识（来自KnowledgeProcessor）
    2. UCB策略（平衡探索与利用）
    3. MCTS思想（树搜索 + 反向传播）
    
    工作流程：
    - 初始化：根据变量重要性构建分层树结构
    - 选择：使用UCB策略选择叶节点
    - 评估：在该叶节点对应的参数空间进行实验
    - 反馈：将实验结果反向传播更新树
    """
    
    def __init__(self, fusion: KnowledgeProcessor, optimization_variables: Dict[str, Any], Cp: float = 5.0, alpha: float = 1.0):
        """
        初始化知识树
        
        Args:
            fusion: KnowledgeProcessor对象，包含化学机理知识和先验分布
            Cp: UCB探索常数，默认2.0（越大越倾向探索）
            alpha: UCB平滑参数，默认1.0（防止除零）
            optimization_variables: 优化变量，格式为 {"vars": [...], "domains": {...}, "var_types": {...}}
        """
        self.fusion = fusion  # 保存知识处理器对象
        self.Cp = Cp  # UCB探索常数
        self.alpha = alpha  # UCB平滑参数
        self._next_id = 0  # 节点ID生成器，从0开始自增
        
        # 使用传入的变量和定义域，或使用全局默认值
        self.vars = optimization_variables.get("vars", [])
        self.domains = optimization_variables.get("domains", {})
        self.var_types = optimization_variables.get("var_types", {})
        
        # 按重要性降序排列变量，重要变量在树的上层优先分裂
        # 只考虑在vars中的变量
        available_vars = [v for v in self.vars if v in fusion.importance]
        order = sorted(available_vars, key=lambda v: -fusion.importance.get(v, 0.0))
        self.order = order  # 保存变量分裂顺序
        
        # 递归构建整棵树，初始box包含所有变量的完整定义域
        self.root = self._build(order, 0, {v: self.domains[v] for v in self.vars if v in self.domains}, parent_id=None)

    def _build(self, order: List[str], idx: int, box: Dict[str, Union[Tuple[float, float], Set[Any]]], parent_id: Optional[int]) -> Node:
        """
        递归构建知识树
        
        按照变量重要性顺序，逐层分裂搜索空间，构建完整的树结构。
        - 对于连续变量：通过区间切分进行分裂
        - 对于离散变量：通过聚类进行分裂
        叶节点被赋予基于化学知识的先验统计量。
        
        Args:
            order: 变量分裂顺序列表（已按重要性排序）
            idx: 当前处理的变量索引
            box: 当前节点的参数空间
                 - 连续变量：(最小值, 最大值) 元组
                 - 离散变量：元素集合
            parent_id: 父节点ID
        
        Returns:
            构建好的Node对象
        """
        # 分配唯一节点ID
        my_id = self._next_id
        self._next_id += 1  # 自增，确保每个节点ID唯一
        
        # 递归终止条件：所有变量都已分裂，到达叶节点
        if idx >= len(order):
            # 创建叶节点（var_name=None表示不再分裂）
            node = Node(node_id=my_id, depth=idx, box=box, var_name=None, parent_id=parent_id)
            
            # 计算该参数区域的先验分数（0到1之间，反映化学合理性）
            s = self.fusion.prior_score_variable_divisions(box)
                    
            # 设置先验访问次数：2-10之间，分数越高先验越强
            node.prior_n = 8.0 * s + 2.0
            
            # 设置先验累积奖励：分数越高，初始期望越高
            # 当s=0时，prior_Q = prior_n * 0（中性）
            # 当s=1时，prior_Q = prior_n * 0.5（乐观）
            node.prior_Q = node.prior_n * (0.5 * s + 0.5)
            
            return node
        
        # 递归情况：继续分裂当前变量
        var = order[idx]  # 获取当前要分裂的变量名
        node = Node(node_id=my_id, depth=idx, box=box, var_name=var, parent_id=parent_id)
        
        if is_continuous(var, self.var_types):
            # 连续变量：通过区间切分进行分裂
            for (lo, hi) in self.fusion.splits.get(var, []):
                # 复制父节点的参数空间
                ch_box = dict(box)
                # 更新当前变量的范围为子区间
                ch_box[var] = (lo, hi)
                # 递归构建子树（深度+1）
                child = self._build(order, idx + 1, ch_box, parent_id=my_id)
                # 添加到子节点列表
                node.children.append(child)
        else:
            # 离散变量：通过聚类进行分裂
            clusters = self.fusion.clusters.get(var, [])
            if not clusters:
                # 如果没有聚类配置，使用整个定义域作为单一聚类
                domain = self.domains.get(var)
                if domain is None:
                    raise ValueError(f"变量 {var} 的定义域为空")
                if isinstance(domain, set):
                    clusters = [domain]
                else:
                    clusters = [set(domain) if hasattr(domain, '__iter__') else set()]
            
            for cluster in clusters:
                if not cluster:  # 跳过空聚类
                    continue
                # 复制父节点的参数空间
                ch_box = dict(box)
                # 更新当前变量的范围为该聚类的元素集合
                ch_box[var] = set(cluster)  # 确保是集合类型
                # 递归构建子树（深度+1）
                child = self._build(order, idx + 1, ch_box, parent_id=my_id)
                # 添加到子节点列表
                node.children.append(child)
        
        return node

    def select_leaf_by_ucb(self, num_leaves: int = 1) -> List[Node]:
        """
        使用UCB策略选择最优叶节点
        
        支持单叶子和多叶子采样：
        - num_leaves = 1: 经典的UCB选择，返回最优的单个叶子。
        - num_leaves > 1: 多叶子采样，用于在优化早期增强探索。
        
        多叶子采样策略:
        1. 计算所有叶子的UCB分数。
        2. 排序并选择Top-k，但加入随机性以避免总是选择相同的组合。
           - 70%确定性选择：选择UCB分数最高的叶子。
           - 30%随机性选择：从分数排名前50%的叶子中随机选择，增加多样性。

        Args:
            num_leaves: 希望选择的叶子数量
            
        Returns:
            选中的叶节点列表
        """
        if num_leaves == 1:
            node = self.root
           # 沿着树向下，每次选择UCB最大的子节点
            while node.children:  # 当前节点不是叶节点
            # 计算父节点的有效访问次数（至少为1，避免log(0)）
                Np = max(1.0, node.n + node.prior_n)
                # 选择UCB值最大的子节点
                node = max(node.children, key=lambda c: c.ucb(Np, self.Cp, self.alpha))
            
            return [node]
        else:
            # 1. 收集所有叶子节点
            leaves = []

            def collect_leaves(n: Node):
                if not n.children:
                    leaves.append(n)
                else:
                    for ch in n.children:
                        collect_leaves(ch)
            collect_leaves(self.root)
            
            if len(leaves) <= num_leaves:
                return leaves
            
            # 2. 计算所有叶子的UCB值
            leaf_ucbs = []
            
            # 创建一个从子节点ID到父节点的映射，用于快速查找
            parent_map = {child.node_id: p for p in self.get_all_nodes() if p.children for child in p.children}
            for leaf in leaves:
                parent = parent_map.get(leaf.node_id)
                if parent:
                    Np = parent.n + parent.prior_n
                    ucb_val = leaf.ucb(Np, self.Cp, self.alpha)
                    leaf_ucbs.append((ucb_val, leaf))

            if not leaf_ucbs:
                return random.sample(leaves, min(num_leaves, len(leaves)))

            # 3. 混合选择策略
            leaf_ucbs.sort(key=lambda x: x[0], reverse=True)
            
            selected = []
            num_deterministic = int(num_leaves * 0.7)
            
            # 70% 确定性选择
            for i in range(min(num_deterministic, len(leaf_ucbs))):
                selected.append(leaf_ucbs[i][1])
            
            # 30% 随机性选择
            num_random = num_leaves - len(selected)
            if num_random > 0:
                # 从Top 50%的候选池中随机选择
                top_half_pool = leaf_ucbs[:max(1, len(leaf_ucbs) // 2)]
                
                # 过滤掉已选中的
                selectable_pool = [item for item in top_half_pool if item[1] not in selected]
                
                if selectable_pool:
                    num_to_sample = min(num_random, len(selectable_pool))
                    random_choices = random.sample(selectable_pool, num_to_sample)
                    selected.extend([item[1] for item in random_choices])

            return selected[:num_leaves]

    def backprop(self, leaf: Node, reward: float):
        """
        反向传播实验奖励到树中
        
        将实验结果（奖励值）沿着从根到叶的路径向上传播。
        
        Args:
            leaf: 进行实验的叶节点
            reward: 奖励值
            alpha_smooth: 平滑系数，用于指数移动平均
        """
        path = []  # 存储从根到叶的路径

        def dfs(cur: Node) -> bool:
            """深度优先搜索找到从根到叶的路径"""
            path.append(cur)  # 将当前节点加入路径
            if cur is leaf:  # 找到目标叶节点
                return True
            for ch in cur.children:  # 递归搜索子节点
                if dfs(ch):
                    return True
            path.pop()  # 回溯：当前路径不通，移除节点
            return False

        # 从根节点开始搜索路径
        dfs(self.root)
        
        # 更新路径上所有节点的统计量
        reward_clipped = float(max(0.0, min(1.0, reward)))
        for nd in path:
            nd.n += 1.0  # 访问次数增加
            nd.Q += reward_clipped

    def get_all_nodes(self) -> List[Node]:
        """返回树中所有节点的列表"""
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        return nodes

    def find_leaf_for_point(self, point_values: Dict[str, Any]) -> Optional[Node]:
        """
        查找包含指定实验参数点的叶节点
        
        给定具体的实验参数值，在树中找到对应的叶节点。用于：
        1. 将历史实验结果映射到树节点进行反向传播
        2. 查询某个参数组合的统计信息
        
        支持连续变量和离散变量：
        - 连续变量：检查值是否在区间内
        - 离散变量：检查值是否在集合中
        
        Args:
            point_values: 参数字典，例如 {"Temperature": 500, "oxygen_pressure": 2.0, ...}
                           键名应与VARS中的变量名一致
        
        Returns:
            包含该点的叶节点，如果点不在搜索空间内则返回None
            
        示例:
            tree.find_leaf_for_point({"Temperature": 500, "oxygen_pressure": 2.0, "Oxidation condition": 0.1})
        """
        def point_in_box(var_name: str, value: Any, box_value: Union[Tuple[float, float], Set[Any]]) -> bool:
            """判断点的某个维度是否在box的对应范围内"""
            if is_continuous(var_name, self.var_types):
                # 连续变量(值全是数字)：检查值是否在区间内
                if isinstance(value, float) or isinstance(value, int):
                    lo, hi = box_value
                    return lo <= value <= hi
                return False
            else:
                # 离散变量：检查值是否在集合中
                if isinstance(value, str):
                    return value in box_value
                return False

        node = self.root  # 从根节点开始
        # 沿着树向下，找到包含该点的子节点
        while node.children:  # 当前节点不是叶节点
            next_node = None
            # 遍历所有子节点，找到包含该点的那个
            for ch in node.children:
                # 检查点的所有维度是否都在子节点的box范围内
                all_match = True
                for var_name in self.vars:
                    if var_name not in point_values.keys():
                        all_match = False
                        break
                    if var_name not in ch.box.keys():
                        all_match = False
                        break
                    if not point_in_box(var_name, point_values[var_name], ch.box[var_name]):
                        all_match = False
                        break
                
                if all_match:  # 所有维度都匹配
                    next_node = ch
                    break  # 找到了，停止搜索
            
            if next_node is None:  # 没有找到包含该点的子节点
                return None  # 该点不在搜索空间内
            
            node = next_node  # 移动到下一层
        
        return node  # 返回包含该点的叶节点

    def export_tree(self) -> Dict[str, Any]:
        """
        导出树的完整结构为可序列化的字典
        
        将整棵知识树导出为JSON兼容的字典格式，包含：
        1. 变量分裂顺序
        2. 每个变量的分割点
        3. 所有节点的详细信息（结构、统计量、UCB参数）
        
        用途：
        - 保存中间结果用于断点续训
        - 可视化树结构和搜索进度
        - 分析和调试优化过程
        
        Returns:
            包含树完整信息的字典，可直接序列化为JSON
        """
        def _node_to_dict(nd: Node, Np: float) -> Dict[str, Any]:
            """将单个节点转换为字典格式"""
            # 转换box为可序列化格式
            serialized_box = {}
            for var_name, box_value in nd.box.items():
                if is_continuous(var_name, self.var_types):
                    # 连续变量：转换元组为列表
                    if isinstance(box_value, tuple):
                        serialized_box[var_name] = list(map(float, box_value))
                    else:
                        serialized_box[var_name] = [0.0, 1.0]  # 默认值
                else:
                    # 离散变量：转换集合为列表
                    if isinstance(box_value, set):
                        serialized_box[var_name] = sorted(list(box_value))
                    else:
                        serialized_box[var_name] = list(box_value) if hasattr(box_value, '__iter__') else []
            
            return {
                "id": nd.node_id,  # 节点唯一标识
                "parent_id": nd.parent_id,  # 父节点ID
                "depth": nd.depth,  # 树深度
                "var_name": nd.var_name,  # 分裂变量名（叶节点为None）
                "box": serialized_box,  # 参数空间范围（已转换为可序列化格式）
                "stats": {  # 节点统计信息
                    "prior_n": nd.prior_n,  # 先验访问次数
                    "prior_Q": nd.prior_Q,  # 先验累积奖励
                    "n": nd.n,  # 实际访问次数
                    "Q": nd.Q,  # 实际累积奖励
                    "ucb": nd.ucb(Np, self.Cp, self.alpha),  # UCB值
                },
                "ucb_hint": None  # UCB计算提示（后续填充）
            }

        nodes = []  # 存储所有节点信息
        stack = [(self.root, 0.0)]  # 栈：(节点, 父节点访问次数)
        
        # 深度优先遍历整棵树
        while stack:
            cur, Np = stack.pop()  # 取出当前节点和父节点访问次数
            
            # 转换节点为字典格式
            d = _node_to_dict(cur, Np)
            
            # 添加UCB计算的上下文信息
            d["ucb_hint"] = {
                "Cp": self.Cp,  # 探索常数
                "alpha": self.alpha,  # 平滑参数
                "parent_visits": float(Np)  # 父节点的有效访问次数
            }
            
            nodes.append(d)  # 添加到节点列表
            
            # 将所有子节点压入栈（附带当前节点的访问次数）
            for ch in cur.children:
                stack.append((ch, max(1.0, cur.n + cur.prior_n)))
        
        # 准备splits和clusters的序列化数据
        splits_info = {}
        clusters_info = {}
        for v in self.vars:
            if is_continuous(v, self.var_types):
                splits_info[v] = self.fusion.splits.get(v, [])
            else:
                # 将聚类中的集合转换为列表以便序列化
                clusters_info[v] = [sorted(list(c)) for c in self.fusion.clusters.get(v, [])]
        
        # 返回完整的树信息
        return {
            "variable_order": self.order,  # 变量分裂顺序（按重要性降序）
            "splits": splits_info,  # 连续变量的分割点列表
            "clusters": clusters_info,  # 离散变量的聚类列表
            "nodes": nodes  # 所有节点的详细信息
        }


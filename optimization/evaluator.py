#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标评估器模块
"""

import math
import json
import importlib
import subprocess
import requests
import concurrent.futures as cf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any 

PENALTY_BAD = 1e-6
EVAL_TIMEOUT_S = 3600
N_WORKERS = 0



def _toy_raw(t: float, p: float, ox: float, domains: Optional[Dict[str, Any]] = None) -> float:
    """玩具评估函数，返回产率（0-100%）"""
    # 尝试获取变量，如果不存在则使用默认值
    temp_domain = domains.get("Temperature", (0, 100))
    press_domain = domains.get("oxygen pressure", (0, 100))
    ox_domain = domains.get("Oxidation condition", (0, 100))
    T0, T1 = temp_domain if isinstance(temp_domain, tuple) and len(temp_domain) == 2 else (0, 100)
    P0, P1 = press_domain if isinstance(press_domain, tuple) and len(press_domain) == 2 else (0, 100)
    O0, O1 = ox_domain if isinstance(ox_domain, tuple) and len(ox_domain) == 2 else (0, 100)
    tn = (t - T0) / (T1 - T0 + 1e-12)
    pn = (p - P0) / (P1 - P0 + 1e-12)
    oxn = (ox - O0) / (O1 - O0 + 1e-12)
    # 计算产率（0-100%）
    yield_val = 50.0 + 40.0 * np.exp(-((tn - 0.6) ** 2) / 0.02) * np.exp(-((pn - 0.25) ** 2) / 0.06) * np.exp(-((oxn - 0.55) ** 2) / 0.05)
    return float(np.clip(yield_val, 0.0, 100.0))


def _call_python_func(py_entry: str, t: float, p: float, ox: float) -> float:
    """调用Python函数，返回产率"""
    mod_name, fn_name = py_entry.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    yield_val = fn(t, p, ox)
    return float(yield_val)


def _call_cli(cli_cmd: str, t: float, p: float, ox: float) -> float:
    """调用命令行工具，返回产率"""
    try:
        cmd = cli_cmd.format(**{"Temperature": t, "oxygen pressure": p, "Oxidation condition": ox})
    except KeyError:
        cmd = cli_cmd.format(T=t, P=p, OX=ox)
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=EVAL_TIMEOUT_S).decode("utf-8")
    data = json.loads(out)
    # 支持 "yield" 或 "y1" 字段
    yield_val = data.get("yield", data.get("y1", 0.0))
    return float(yield_val)


def _call_http(http_url: str, t: float, p: float, ox: float) -> float:
    """调用HTTP API，返回产率"""
    r = requests.post(http_url, json={"Temperature": t, "oxygen pressure": p, "Oxidation condition": ox},
                      timeout=EVAL_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    # 支持 "yield" 或 "y1" 字段
    yield_val = data.get("yield", data.get("y1", 0.0))
    return float(yield_val)


def _load_excel_data(excel_path: str) -> pd.DataFrame:
    """
    加载Excel文件中的实验数据
    
    参数:
        excel_path: Excel文件路径
        
    返回:
        DataFrame或None
    """
    try:
        df = pd.read_excel(excel_path, keep_default_na=False)
            # 将df中的所有元素末尾的空格去掉
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        # 将df中的所有元素开头的空格去掉
        df = df.map(lambda x: x.lstrip() if isinstance(x, str) else x)
    except Exception as e:
        print(f"[evaluator] 警告：无法读取Excel文件 {excel_path}: {e}")
        return None
    return df


def _find_matching_experiment(df: pd.DataFrame, 
                               reaction_conditions: Dict[str, Any],
                               reactants: Dict[str, Any],
                               targets: Dict[str, Any],
                               tolerance: float = 1e-2) -> Optional[Dict[str, float]]:
    """
    在CSV数据中查找匹配的实验数据
    
    参数:
        df: 包含实验数据的DataFrame
        reaction_conditions: 反应条件字典，包含变量
        reactants: 反应物信息字典
        targets: 目标值信息字典
        tolerance: 数值匹配的容差（用于连续变量）
        
    返回:
        如果找到匹配的实验，返回目标值字典，否则返回None
    """
    if df.empty:
        return None
    
    # 构建匹配条件
    mask = pd.Series([True] * len(df))
    
    # 匹配反应条件（连续变量使用容差，离散变量精确匹配）
    eval_vars = reaction_conditions.keys()
    for var_name in eval_vars:
        value = reaction_conditions[var_name]
        # 对于连续变量，使用容差匹配
        if isinstance(value, (int, float)):
            mask = mask & ((df[var_name] - value).abs() <= tolerance)
        else:
            # 对于离散变量，精确匹配
            mask = mask & (df[var_name] == value)
    
    # 匹配反应物（如果CSV中有反应物列）
    for reactant_key, reactant_value in reactants.items():
        if reactant_key in df.columns:
            mask = mask & (df[reactant_key] == reactant_value)
    
    # 查找匹配的行
    matched_rows = df[mask]
    
    if matched_rows.empty:
        return {}
    
    # 取每一行匹配的数据
    rows = [matched_rows.iloc[i] for i in range(len(matched_rows))]
    targets_val = {target_key: float('-inf') for target_key in targets.keys()}
    for row in rows:
        for target_key in targets.keys():
            targets_val[target_key] = max(targets_val[target_key], float(row[target_key]))
    return targets_val

class Evaluator:
    """
    评估器类，用于评估不同模式下的反应条件
    """
    def __init__(self,
                 eval_mode: str,
                 excel_path: Optional[str] = None,
                 reactants: Optional[Dict[str, Any]] = None,
                 cli_cmd: Optional[str] = None,
                 http_url: Optional[str] = None,
                 py_entry: Optional[str] = None):
        self.eval_mode = eval_mode
        self.reactants = reactants
        self.cli_cmd = cli_cmd
        self.http_url = http_url
        self.py_entry = py_entry
        self.excel_df = None
        
        if self.eval_mode == "excel":
            if not excel_path:
                raise ValueError("eval_mode='excel' 需要提供 excel_path 参数")
            self.excel_df = _load_excel_data(excel_path)
            if self.excel_df.empty:
                print("[evaluator] 警告：Excel数据为空，将返回默认惩罚值")

    def _evaluate_one_point(self, reaction_conditions: Dict[str, Any], targets: Dict[str, Any], ranges: Dict[str, Any]) -> Dict[str, float]:
        """
        评估单个反应条件的产率
        
        参数:
            reaction_conditions: 反应条件字典
            targets: 目标值字典
            
        返回:
            目标值字典
        """
        result_bad = {}
        try:
            if self.eval_mode == "excel":
                if self.excel_df is None or self.excel_df.empty:
                    for target_key in targets.keys():
                        if targets[target_key] == "minimize":
                            result_bad[target_key] = ranges[target_key][1]
                        else:
                            result_bad[target_key] = ranges[target_key][0]
                    return result_bad
                
                targets_val = _find_matching_experiment(self.excel_df, reaction_conditions, self.reactants or {}, targets=targets)
                if targets_val is {} or not targets_val:
                    for target_key in targets.keys():
                        if targets[target_key] == "minimize":
                            result_bad[target_key] = ranges[target_key][1]
                        else:
                            result_bad[target_key] = ranges[target_key][0]
                    return result_bad
                return targets_val

            # 对于其他模式，提取通用变量并调用相应函数
            t = reaction_conditions.get("Temperature", 0.0)
            p = reaction_conditions.get("oxygen pressure", 0.0)
            ox = reaction_conditions.get("Oxidation condition", 0.0)
            
            yield_val = 0.0

            if self.eval_mode == "python":
                if not self.py_entry:
                    raise ValueError("eval_mode='python' 需要提供 py_entry 参数")
                yield_val = _call_python_func(self.py_entry, t, p, ox)
            elif self.eval_mode == "cli":
                if not self.cli_cmd:
                    raise ValueError("eval_mode='cli' 需要提供 cli_cmd 参数")
                yield_val = _call_cli(self.cli_cmd, t, p, ox)
            elif self.eval_mode == "http":
                if not self.http_url:
                    raise ValueError("eval_mode='http' 需要提供 http_url 参数")
                yield_val = _call_http(self.http_url, t, p, ox)
            else:
                raise ValueError(f"不支持的评估模式: {self.eval_mode}")

            # 假设单一返回值对应于所有目标键
            return {target_key: yield_val for target_key in targets.keys()}

        except Exception as e:
            print(f"[evaluator] 评估失败: {e}")
            for target_key in targets.keys():
                if targets[target_key] == "minimize":
                    result_bad[target_key] = ranges[target_key][1]
                else:
                    result_bad[target_key] = ranges[target_key][0]
            return result_bad

    def evaluate_batch(self, X_batch: List[Dict[str, Any]], targets: Dict[str, Any], ranges: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        批量评估反应条件
        
        参数:
            X_batch: 反应条件字典列表
            targets: 目标值字典
            
        返回:
            目标值字典列表
        """
        if N_WORKERS and N_WORKERS > 0:
            outs = [None] * len(X_batch)
            with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
                futs = {ex.submit(self._evaluate_one_point, X_batch[i], targets): i for i in range(len(X_batch))}
                for fut in cf.as_completed(futs):
                    idx = futs[fut]
                    try:
                        outs[idx] = fut.result(timeout=EVAL_TIMEOUT_S)
                    except Exception:
                        result_bad = {}
                        for target_key in targets.keys():
                            if targets[target_key] == "minimize":
                                result_bad[target_key] = ranges[target_key][1]
                            else:
                                result_bad[target_key] = ranges[target_key][0]
                        outs[idx] = result_bad
            return outs
        else:
            return [self._evaluate_one_point(x, targets, ranges) for x in X_batch]


def make_evaluator(eval_mode: str,
                   excel_path: Optional[str] = None,
                   reactants: Optional[Dict[str, Any]] = None,
                   cli_cmd: Optional[str] = None,
                   http_url: Optional[str] = None,
                   py_entry: Optional[str] = None):
    """
    创建评估器函数
    
    参数:
        eval_mode: 评估模式 ("toy", "python", "cli", "http", "excel")
        excel_path: Excel文件路径（eval_mode="excel"时使用）
        reactants: 反应物信息字典（eval_mode="excel"时使用）
        cli_cmd: 命令行模板（eval_mode="cli"时使用）
        http_url: HTTP API URL（eval_mode="http"时使用）
        py_entry: Python入口函数（eval_mode="python"时使用）
    """
    evaluator = Evaluator(
        eval_mode=eval_mode,
        excel_path=excel_path,
        reactants=reactants,
        cli_cmd=cli_cmd,
        http_url=http_url,
        py_entry=py_entry,
    )
    return evaluator


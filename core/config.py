#!/usr/-bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置模型
使用 Pydantic 进行数据验证和类型提示
"""
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import os
from omegaconf import OmegaConf

# --- 子配置模型 ---

class LLMConfig(BaseModel):
    """LLM 相关配置"""
    base_url: Optional[str] = Field(None, description="LLM API 的基础 URL")
    api_key: Optional[str] = Field(None, description="LLM API 密钥")
    model_name: str = Field("gemini-2.5-pro", description="要使用的 LLM 模型名称")
    max_retries: int = Field(5, description="最大重试次数")
    retry_delay: int = Field(10, description="重试延迟")

class PathsConfig(BaseModel):
    """路径相关配置"""
    outdir: str = Field("results", description="主输出目录")
    logdir: str = Field("logs", description="日志目录")
    experiment_process_path: Optional[str] = Field(None, description="实验步骤路径")
    search_results_path: Optional[str] = Field(None, description="文献检索结果路径")
    schema_path: Optional[str] = Field(None, description="Schema 路径")
    mechanism_structured_data_path: Optional[str] = Field(None, description="结构化机理数据路径")
    variables_structured_data_path: Optional[str] = Field(None, description="结构化变量数据路径")
    optimization_variables_path: Optional[str] = Field(None, description="优化变量路径")
    refined_optimization_variables_path: Optional[str] = Field(None, description="精炼后的优化变量路径")
    variable_divisions_path: Optional[str] = Field(None, description="变量划分路径")

class KnowledgeConfig(BaseModel):
    """知识树相关配置"""
    Cp: float = Field(10.0, description="UCB探索常数")
    alpha: float = Field(1.0, description="UCB平滑参数")
    use_history_data: bool = Field(False, description="是否使用历史数据进行知识树更新")
    history_csv_path: Optional[str] = Field(None, description="历史数据保存路径")

# --- 主配置模型 ---

class AppConfig(BaseModel):
    """项目主配置"""
    project_name: str = "OptimizerPipeline"
    seed: Optional[int] = Field(None, description="随机种子")
    
    paths: PathsConfig
    llm: LLMConfig
    knowledge: KnowledgeConfig
    agents: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent configurations")
    
    # Runtime parameters that can be injected via config
    reactants: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Reaction components",)
    search_query: Optional[str] = Field(None, description="Search query for literature")
    targets: Optional[Dict[str, Any]] = Field(None, description="目标变量")

    @classmethod
    def from_omegaconf(cls, conf: Any) -> 'AppConfig':
        """
        Validates and creates AppConfig from an OmegaConf DictConfig object.
        """
        # Resolve all variables (e.g. ${oc.env:VAR})
        OmegaConf.resolve(conf)
        
        # Convert to dictionary
        config_dict = OmegaConf.to_container(conf, resolve=True)
        
        # Use Pydantic for validation
        return cls.model_validate(config_dict)

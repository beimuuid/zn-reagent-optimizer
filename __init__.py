#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识引导多目标贝叶斯优化系统

这是一个模块化的优化系统，包含以下主要组件：

模块结构：
- core: 核心流程模块（pipeline, main）
- knowledge: 知识管理模块（knowledge_tree, knowledge_process）  
- optimization: 优化器模块（leaf_mobo）
- agents: 代理模块（schema_generation, variable_extraction, knowledge_refinement, structured_data_extraction, literature_extraction）
- utils: 工具模块（gemini_api_simple, json_to_csv, json_to_pdf, common utilities, pdf_utils, pipeline_utils）
"""

# 核心流程
from .core import run_pipeline, run_from_config, AppConfig

# 知识管理
from .knowledge import (
    KnowledgeProcessor,
    KnowledgeTree,
    Node
)
# 代理
from .agents import (
    SchemaGenerationAgent,
    VariableExtractionAgent,
    KnowledgeRefinementAgent,
    StructuredDataExtractionAgent,
    LiteratureExtractionAgent,
)

# 优化器
from .optimization import (
    LeafMOBO,
    HAVE_BOTORCH,
    make_evaluator
)

# 工具函数
from .utils import (
    aggregate_reward_from_batch,
    call_gemini_api,
    json_to_csv_schema,
    json_to_pdf,
    set_seed,
    is_continuous,
    is_discrete,
    safe_eval_expr,
    almost_equal_tuple,
    short,
    _looks_like_evidence_item,
    _rough_range_from_text,
    extract_title_from_pdf,
    setup_logger,
    make_json_serializable,
    load_history_csv,
)


__version__ = "2.0.0"

__all__ = [
    # 核心流程
    "run_pipeline",
    "run_from_config",
    "AppConfig",
    
    # 代理
    "SchemaGenerationAgent",
    "VariableExtractionAgent",
    "KnowledgeRefinementAgent",
    "StructuredDataExtractionAgent",
    "LiteratureExtractionAgent",
    
    # 知识管理
    "KnowledgeProcessor",
    "KnowledgeTree",
    "Node",
    
    # 优化器
    "LeafMOBO",
    "HAVE_BOTORCH",
    "make_evaluator",
    
    # 工具函数
    "aggregate_reward_from_batch",
    "call_gemini_api",
    "json_to_csv_schema",
    "json_to_pdf",
    "set_seed",
    "is_continuous",
    "is_discrete",
    "safe_eval_expr",
    "almost_equal_tuple",
    "short",
    "_looks_like_evidence_item",
    "_rough_range_from_text",
    "extract_title_from_pdf",
    "setup_logger",
    "make_json_serializable",
    "load_history_csv",
]

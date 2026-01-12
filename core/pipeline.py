#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主优化流程模
"""

import sys
import os
import json
import tempfile
import pathlib
import pandas as pd
import random
import logging
import asyncio
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from collections import defaultdict
import numpy as np

from .config import AppConfig
from .step import (
    step_schema_generation,
    step_variables_extraction,
    step_variables_refinement,
    step_variable_divisions_extraction
)
from tools.extract_from_local_pdf import extract_data_async
from utils import setup_logger, set_seed
from agents import (
    SchemaGenerationAgent,
    KnowledgeRefinementAgent,
    VariableExtractionAgent,
)


def run_pipeline(cfg: AppConfig,
                 reactants: Dict[str, Any],
                 search_query: Optional[str] = None):
    """
    运行完整的优化流程
    """
    if cfg.seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = random.randint(1, 1000000)
        set_seed(cfg.seed)
    logger = setup_logger(cfg.paths.logdir)
    if search_query:
        cfg.search_query = search_query
    if reactants:
        cfg.reactants = reactants


    # --- 初始化 ---
    logger.info("=" * 60)
    logger.info(f"随机种子: {cfg.seed}")
    logger.info(f"反应物: {cfg.reactants}")
    logger.info(f"检索查询: {cfg.search_query}")
    logger.info("=" * 60)

    # 创建时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_outdir = os.path.join(cfg.paths.outdir, f"run_{timestamp}")
    os.makedirs(timestamped_outdir, exist_ok=True)
    logger.info(f"所有输出将保存到: {timestamped_outdir}")

    # --- Agent 初始化 ---
    logger.info("\n[Initialization] 正在初始化 Agents...")
    schema_generation_agent = SchemaGenerationAgent(cfg.agents["schema_generation"])
    variable_extraction_agent = VariableExtractionAgent(cfg.agents["variable_extraction"])
    knowledge_refinement_agent = KnowledgeRefinementAgent(cfg.agents["knowledge_refinement"])
    logger.info("  * 所有 Agents 初始化完成")
    
    # --- 流程步骤 ---

    additional_info = """
    专家关注的关键字为：有机锌试剂结构(organozinc_reagent_structure)，溶剂(solvent)，添加剂(additive)，催化剂(catalyst)。
    """
    # 2. Schema 生成
    schema_csv_path = step_schema_generation(cfg, schema_generation_agent, cfg.search_query, timestamped_outdir, additional_info, logger)
    if not schema_csv_path:
        logger.error("Schema 生成步骤失败，流程终止。")
        return  

    mechanism_structured_data, variables_structured_data = asyncio.run(extract_data_async(
        cfg=cfg,
        pdf_path="/root/optimizer/init_files/zn/pdfs",
        output_dir="/root/optimizer/init_files/zn",
        schema_csv_path=schema_csv_path
    ))

    # 4. 变量提取
    optimization_variables = step_variables_extraction(cfg, variable_extraction_agent, variables_structured_data, cfg.optimization.targets, timestamped_outdir, logger)
    if not optimization_variables:
        logger.error("变量提取步骤失败，流程终止。")
        return
    
    additional_context_variables = """
    专家关注的变量为：有机锌试剂（organozinc_reagent），溶剂（solvent）添加剂（additive），催化剂（catalyst）。
    请针对这四个变量，从已经提取到的优化变量里面精炼出对应变量的优化空间和变量类型。不要添加任何其他变量或空间。
    """
  
    # 5. 变量精炼
    refined_optimization_variables = step_variables_refinement(cfg, knowledge_refinement_agent, optimization_variables, additional_context=additional_context_variables, output_dir=timestamped_outdir, logger=logger)
    if not refined_optimization_variables:
        logger.error("变量精炼步骤失败，流程终止。")
        return
    
    # 6. 变量划分提取
    variable_divisions = step_variable_divisions_extraction(cfg, variable_extraction_agent, mechanism_structured_data, refined_optimization_variables, output_dir=timestamped_outdir, logger=logger)
    if not variable_divisions:
        logger.error("变量划分提取步骤失败，流程终止。")
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("流程执行完毕")
    logger.info("=" * 60)



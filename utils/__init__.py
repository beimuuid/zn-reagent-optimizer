#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
"""
from .logging import setup_logger
from .gemini_api_simple import call_gemini_api
from .json_to_csv import json_to_csv_schema
from .acq import optimize_acqf_discrete_idx
from .reward import aggregate_reward_from_batch
from .pdf import extract_title_from_pdf
from .common import (
    set_seed,
    make_json_serializable,
    load_history_csv,
    is_continuous,
    is_discrete,
    safe_eval_expr,
    almost_equal_tuple,
    short,
    _looks_like_evidence_item,
    _rough_range_from_text,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "setup_logger",
    "call_gemini_api",
    "json_to_csv_schema",
    "optimize_acqf_discrete_idx",
    "aggregate_reward_from_batch",
    "extract_title_from_pdf",
    "make_json_serializable",
    "load_history_csv",
    "set_seed",
    "is_continuous",
    "is_discrete",
    "safe_eval_expr",
    "almost_equal_tuple",
    "short",
    "_looks_like_evidence_item",
    "_rough_range_from_text",
    "save_checkpoint",
    "load_checkpoint",
]


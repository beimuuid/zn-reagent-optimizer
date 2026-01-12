#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口模块
"""

import hydra
from .config import AppConfig
from .pipeline import run_pipeline
import time

def run_with_config(config_path: str, config_name: str):
    """用指定的配置路径和名称运行 pipeline"""
    # 使用 context manager 确保 GlobalHydra 状态被正确清理
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name)
        # Validate config using Pydantic
        app_config = AppConfig.from_omegaconf(cfg)
        
        print(f"\n>>> 正在使用配置路径 [{config_path}] 运行 pipeline...")
        run_pipeline(
            cfg=app_config,
            reactants=app_config.reactants,
            search_query=app_config.search_query
        )


def main():
    # 定义要加载的两个不同配置路径
    config_paths = ["../configs"]
    config_name = "config"
    
    for path in config_paths:
        try:
            run_with_config(path, config_name)
            # 睡眠10秒
            time.sleep(5)
        except Exception as e:
            print(f"在运行配置路径 {path} 时出错: {e}")


if __name__ == "__main__":
    main()

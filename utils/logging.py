import logging
import os
import json
import pathlib
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

try:
    import pandas as pd
except ImportError:
    pd = None


def setup_logger(outdir: str) -> logging.Logger:
    """
    配置日志记录器，将日志保存到文件并输出到控制台
    """
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(outdir, f"pipeline_{timestamp}.log")
    
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件已创建: {log_file}")
    return logger
import argparse
import os
import json
from pathlib import Path
import asyncio
import re
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AppConfig
from agents import StructuredDataExtractionAgent
from utils import extract_title_from_pdf


# ----------------------------------------------------------------------------------
# 重用项目中的PDF解析代码 (来自 optimizer/agents/literature_extraction.py)
# ----------------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

class PDFParser:
    """使用 PyMuPDF 解析 PDF 文件"""
    def __init__(self):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) 未安装, 请运行 `pip install PyMuPDF`")

    async def parse(self, pdf_path: str) -> str:
        """异步解析 PDF 文件为文本"""
        return await asyncio.to_thread(self._parse_sync, pdf_path)

    def _parse_sync(self, pdf_path: str) -> str:
        """同步解析实现"""
        doc = fitz.open(pdf_path)
        return "".join(page.get_text("text") for page in doc)

# ----------------------------------------------------------------------------------

async def extract_data_async(cfg: AppConfig, pdf_path: str, output_dir: str, schema_csv_path: str):
    """
    异步执行数据提取的核心逻辑
    """
    #提取路径下的所有pdf文件
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]

    mock_search_results = []
    data_extraction_agent = StructuredDataExtractionAgent(cfg.agents["structured_data_extraction"])
    pdf_parser = PDFParser()
    print("✅ Agents 初始化完成。")
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_file_path = os.path.join(pdf_path, pdf_file)
        final_schema_path = schema_csv_path
        try:
            markdown_content = await pdf_parser.parse(pdf_file_path)
            if not markdown_content.strip():
                print("[错误] PDF内容为空或解析失败。")
                print(f"PDF文件路径: {pdf_file_path}")
                continue
        except Exception as e:
            print(f"[错误] 解析PDF时发生错误: {e}")
            return

        # 从pdf文件名中提取标题
        paper_title = extract_title_from_pdf(pdf_file_path)
        doi = pdf_file.replace(".pdf", "").replace("_", "/")
        mock_search_results.append({
            "title": paper_title,
            "doi": doi,
            "pdf_path": pdf_file_path,
            "markdown_content": markdown_content
        })
        if len(mock_search_results) >= 300:
            break

    with open(os.path.join(output_dir, "search_results.json"), 'w', encoding='utf-8') as f:
        json.dump(mock_search_results, f, indent=2, ensure_ascii=False)
    print("✅ 搜索结果保存完成。")
    # 提取结构化数据
    mechanism_data, variables_data = data_extraction_agent.extract_from_search_results(
        search_results=mock_search_results,
        schema_csv_path=final_schema_path
    )
    # print("✅ 结构化数据提取完成。")
    classified_papers = data_extraction_agent.classify_papers(
        papers=mock_search_results,
    )
    print("✅ 文献分类完成。")

    # 保存结果
    with open(os.path.join(output_dir, "mechanism_structured_data.json"), 'w', encoding='utf-8') as f:
        json.dump(mechanism_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "variables_structured_data.json"), 'w', encoding='utf-8') as f:
        json.dump(variables_data, f, indent=2, ensure_ascii=False)
    
    # 去掉extracted_data字段
    for item in mechanism_data:
        try:
            item.pop("extracted_data")
        except KeyError:
            print(f"KeyError: {item}")
            continue
    with open(os.path.join(output_dir, "mechanism_structured_data_simple.json"), 'w', encoding='utf-8') as f:
        json.dump(mechanism_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "classified_papers.json"), 'w', encoding='utf-8') as f:
        json.dump(classified_papers, f, indent=2, ensure_ascii=False)
    print("✅ 文献分类结果保存完成。")
    return mechanism_data, variables_data

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    主函数入口，由 Hydra 负责加载和管理配置。
    """
    # --- 1. 验证并转换配置 ---
    app_config = AppConfig.from_omegaconf(cfg)

    # --- 2. 从命令行覆盖参数 ---
    # Hydra 允许通过命令行直接覆盖配置项
    # 例如：python extract_from_local_pdf.py pdf_path=path/to/my.pdf
    pdf_path = cfg.get("pdf_path", "/root/optimizer/init_files/zn/pdfs")
    schema_path = cfg.get("schema_path", "/root/optimizer/init_files/zn/final_schema.csv")
    output_dir = "/root/optimizer/init_files/zn"

    if not schema_path:
        print("错误: 必须提供 schema_path 参数。")
        print("请在命令行中指定: python extract_from_local_pdf.py schema_path=path/to/your/schema.csv")
        return

    # --- 3. 运行异步任务 ---
    mechanism_data, variables_data = asyncio.run(extract_data_async(
        cfg=app_config,
        pdf_path=pdf_path,
        output_dir=output_dir,
        schema_csv_path=schema_path
    ))
    print("\n🎉 流程执行完毕。")


if __name__ == "__main__":
    main()

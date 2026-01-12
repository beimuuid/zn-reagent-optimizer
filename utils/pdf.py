import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def extract_title_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 元数据中提取标题，如果失败则返回文件名
    """
    if not HAS_PYMUPDF:
        return Path(pdf_path).stem

    try:
        doc = fitz.open(pdf_path)
        title = doc.metadata.get('title')
        if title:
            return title
    except Exception as e:
        print(f"[警告] 读取 PDF 元数据失败: {pdf_path}, 错误: {e}")
    
    # 如果无法从元数据获取标题，则返回文件名（不含扩展名）
    return Path(pdf_path).stem

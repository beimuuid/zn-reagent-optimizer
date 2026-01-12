import sys
import os
import asyncio
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import datetime
import requests
import paramiko
import json
from scp import SCPClient

from utils import extract_title_from_pdf

# 尝试导入PDF处理库
try:
    from unstructured_client import UnstructuredClient
    from unstructured_client.models import shared
    from unstructured_client.models.errors import SDKError
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class PDFParser:
    """使用 PyMuPDF 解析 PDF 文件"""
    def __init__(self):
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) 未安装，请运行 `pip install PyMuPDF`")

    async def parse(self, pdf_path: str) -> str:
        return await asyncio.to_thread(self._parse_sync, pdf_path)

    def _parse_sync(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text("text") for page in doc)

class ChunkrParser:
    """使用 Unstructured API (Chunkr) 解析 PDF 文件"""
    def __init__(self):
        if not HAS_UNSTRUCTURED:
            raise ImportError("Unstructured 客户端未安装，请运行 `pip install unstructured-client`")
        api_key = os.getenv("CHUNKR_API_KEY")
        if not api_key:
            raise ValueError("环境变量 CHUNKR_API_KEY 未设置")
        self.client = UnstructuredClient(api_key=api_key)

    async def parse(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            files = shared.Files(content=f.read(), file_name=os.path.basename(pdf_path))
        req = shared.PartitionParameters(files=files, strategy="hi_res")
        try:
            res = self.client.general.partition(req)
            return "\n\n".join([str(el) for el in res.elements])
        except SDKError as e:
            print(f"[错误] 使用 Unstructured API 解析失败: {e}")
            return ""


class LiteratureExtractionAgent:
    """
    通过调用服务器API进行文献检索、下载和内容提取的 Agent
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.server_url = self.cfg.get("server_url", "http://localhost:6666/dr")
        self.timeout = self.cfg.get("timeout_seconds", 600)
        self.scp_cfg = self.cfg.get("scp_config")
        self.request_template = self.cfg.get("request_template", {})
        
        # 本地缓存目录，用于存放SCP下载的PDF
        self._optimizer_root = Path(__file__).resolve().parent.parent.parent
        self.local_pdf_cache = self._optimizer_root / ".cache" / "pdfs"
        self.local_pdf_cache.mkdir(parents=True, exist_ok=True)
        self.local_json_cache = self._optimizer_root / ".cache" / "jsons"
        self.local_json_cache.mkdir(parents=True, exist_ok=True)
        
        self.parser = self._setup_parser()

    def _setup_parser(self):
        parser_type = self.cfg.get('pdf_parser', 'pymupdf')
        if parser_type == "pymupdf":
            return PDFParser()
        elif parser_type == "chunkr":
            return ChunkrParser()
        else:
            raise ValueError(f"不支持的 PDF 解析器类型: {parser_type}")

    def _create_scp_client(self):
        """创建并返回一个 SCP 客户端"""
        if not self.scp_cfg:
            raise ValueError("Agent 配置中缺少 'scp_config' 部分")
        
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.scp_cfg["host"],
            port=self.scp_cfg.get("port", 22),
            username=self.scp_cfg["username"],
            password=self.scp_cfg["password"]
        )
        return SCPClient(client.get_transport())

    async def _process_successful_response(self, query: str, res: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理成功的 API 响应，下载并解析相关的 PDF 文件"""
        results = []
        try:
            # 1. 下载 JSON 结果文件
            remote_paths = res.get('last_query_pdfs', [])
            if len(remote_paths) < 2:
                print(f"[警告] API 响应缺少预期的 JSON 文件路径, query: {query}")
                return []

            remote_xinghe_json_path, remote_web_json_path = remote_paths[0], remote_paths[1]
            try:
                with self._create_scp_client() as scp:
                    scp.get(remote_xinghe_json_path, local_path=str(self.local_json_cache))
                xinghe_json_path = self.local_json_cache / Path(remote_xinghe_json_path).name
            except Exception as e:
                print("web json file not found")
                web_json_path = None
            try:
                with self._create_scp_client() as scp:
                    scp.get(remote_web_json_path, local_path=str(self.local_json_cache))
                web_json_path = self.local_json_cache / Path(remote_web_json_path).name
            except Exception as e:
                print("web json file not found")
                web_json_path = None
            if xinghe_json_path is None and web_json_path is None:
                print(f"[错误] 下载 JSON 文件失败: {e}")
                return []
            if xinghe_json_path is not None:
                with open(xinghe_json_path, 'r', encoding='utf-8') as f:
                    xinghe_json = json.load(f)
            if web_json_path is not None:
                with open(web_json_path, 'r', encoding='utf-8') as f:
                    web_json = json.load(f)

            # 3. 下载 PDF 文件
            pdf_paths_to_download = xinghe_json.get('last_downloaded_path', []) if xinghe_json_path is not None else [] + web_json.get('last_downloaded_path', []) if web_json_path is not None else []
            
            if not pdf_paths_to_download:
                print(f"[信息] 未在 JSON 文件中找到 PDF 路径, query: {query}")
                return []

            with self._create_scp_client() as scp:
                for remote_pdf_path in pdf_paths_to_download:
                    remote_full_pdf_path = self.scp_cfg['remote_path_root'] + "/" + remote_pdf_path
                    scp.get(remote_full_pdf_path, local_path=str(self.local_pdf_cache))

            # 4. 解析 PDF 并收集结果
            for remote_pdf_path in pdf_paths_to_download:
                local_pdf_path = self.local_pdf_cache / Path(remote_pdf_path).name
                if local_pdf_path.exists():
                    try:
                        title = extract_title_from_pdf(str(local_pdf_path))
                        markdown_content = await self.parser.parse(str(local_pdf_path))
                        results.append({
                            "query": query,
                            "title": title or local_pdf_path.stem,
                            "pdf_path": str(local_pdf_path),
                            "markdown_content": markdown_content,
                        })
                    except Exception as e:
                        print(f"[错误] 处理本地 PDF 文件 {local_pdf_path} 失败: {e}")
                else:
                    print(f"[警告] 未找到预期的本地 PDF 文件: {local_pdf_path}")

        except Exception as e:
            print(f"[错误] 处理成功的响应时发生异常, query: {query}, error: {e}")

        return results

    async def search_and_extract(
        self,
        query: str,
        max_results_per_query: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        payload = []
        item = self.request_template.copy()
        item["question"] = item.get("question", "{question}").format(
            question=query, max_results=max_results_per_query
        )
        item["xinghe_cache_dir"] = item.get("xinghe_cache_dir", "cache_chem_001")
        item["web_cache_dir"] = item.get("web_cache_dir", "web_chem_001")
        item["base_model"] = item.get("base_model", "gemini-2.5-pro")
        payload.append(item)

        all_results = []
        try:
            print(f"[信息] 向 {self.server_url} 发送包含 {len(payload)} 个任务的批量请求...")
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                print(f"[信息] 收到 API 响应。完成任务数: {data.get('count', 'N/A')}")
                
                for i, res in enumerate(data.get('results', [])):
                    if res.get('status') == 'success':
                        processed_results = await self._process_successful_response(query, res)
                        all_results.extend(processed_results)
                        print(f"[信息] 查询 '{query}' 的任务成功。")
                    else:
                        print(f"[错误] 查询 '{query}' 的任务失败: {res.get('error', '未知错误')}")
            else:
                print(f"[错误] 请求失败，状态码: {response.status_code}, 内容: {response.text}")

        except Exception as e:
            print(f"[错误] 调用 API 或处理过程中发生异常: {e}")
        
        print(f"[完成] 共处理 {len(all_results)} 篇文献")
        return all_results

    def search_and_extract_sync(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return asyncio.run(self.search_and_extract(*args, **kwargs))


    def search_for_literature(self, query: str) -> List[Dict[str, Any]]:
        if not query:
            return []

        payload = []
        all_results = []
        item = self.request_template.copy()
        item["question"] = item.get("question", "{question}").format(
            question=query
        )
        item["xinghe_cache_dir"] = item.get("xinghe_cache_dir", "cache_chem_001")
        item["web_cache_dir"] = item.get("web_cache_dir", "web_chem_001")
        item["base_model"] = item.get("base_model", "gemini-2.5-pro")
        payload.append(item)
        response = requests.post(self.server_url, json=payload, timeout=self.timeout)
        if response.status_code == 200:
            data = response.json()
            print(f"[信息] 收到 API 响应。完成任务数: {data.get('count', 'N/A')}")
            for res in data.get('results', []):
                if res.get('status') == 'success':
                    processed_results = res.get('result',"[]")
                    all_results.extend(json.loads(processed_results))
                else:
                    print(f"[错误] 查询 '{query}' 的任务失败: {res.get('error', '未知错误')}")
            return all_results
        else:
            print(f"[错误] 请求失败，状态码: {response.status_code}, 内容: {response.text}")
            return []
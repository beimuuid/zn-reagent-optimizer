import json
import csv
import pathlib
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import pandas as pd
import concurrent.futures

from .base import BaseKnowledgeAgent

class StructuredDataExtractionAgent(BaseKnowledgeAgent):
    """
    Agent for extracting structured data from documents based on a schema.
    """

    def _process_schema_for_prompt(self, file_path: str) -> Tuple[str, str, str]:
        """
        Reads the CSV Schema file, classifies fields by level, and
        generates structured text descriptions for the LLM prompt.
        """
        base_info_lines, process_lines, mechanism_lines = [], [], []
        col_name_en = '字段名（英文）'
        col_name_cn = '字段名（中文）'
        col_desc = '完整描述'
        col_level = '层级'
        col_hint = '提取提示'
        
        try:
            with open(file_path, mode='r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # 获取关键信息，并去除前后空格
                    field_en = row.get(col_name_en, '').strip()
                    field_cn = row.get(col_name_cn, '').strip()
                    desc = row.get(col_desc, '').strip()
                    level = row.get(col_level, '').strip()
                    hint = row.get(col_hint, '').strip()
                    
                    # 如果缺少关键字段名，跳过该行
                    if not field_en:
                        continue

                    # 格式化单行描述文本
                    # 格式： - **Field_Name** (中文名): 描述 [提示: ...]
                    formatted_line = f"- **{field_en}** ({field_cn}): {desc}"
                    if hint:
                        formatted_line += f" [提示: {hint}]"

                    # 根据 Level 进行分类
                    if 'Level 1' in level:
                        base_info_lines.append(formatted_line)
                    elif 'Level 2' in level or 'Level 3' in level:
                        process_lines.append(formatted_line)
                    elif 'Level 4' in level:
                        mechanism_lines.append(formatted_line)
                    else:
                        # 如果有未标记层级的字段，默认放入基础信息
                        base_info_lines.append(formatted_line)

        except FileNotFoundError:
            raise FileNotFoundError(f"Schema CSV file not found at: {file_path}")
        except Exception as e:
            raise IOError(f"Error reading schema CSV: {e}")

        process_text = "### 2. Reaction Variables and Evaluation Metrics Material\n" + "\n".join(process_lines)
        mechanism_text = "### 3. Mechanism and Optimization Logic Material\n" + "\n".join(mechanism_lines)
        
        # Base text is not used in the new prompts, returning empty string for compatibility
        base_text = "" 

        return base_text, process_text, mechanism_text

    def _extract_for_mechanism(self, markdown_content: str, process_text: str, mechanism_text: str, paper_title: str) -> List[Dict[str, Any]]:
        """Calls LLM to extract mechanism-related information."""
        prompt = self.get_prompt(
            'extract_mechanism',
            paper_title=paper_title,
            process_text=process_text,
            mechanism_text=mechanism_text,
            markdown_content=markdown_content
        )
        system_prompt = "You are a professional literature information extraction expert. Please strictly follow the requirements to extract information, and only output valid JSON format."
        return self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or []

    def _extract_for_variables(self, markdown_content: str, process_text: str, paper_title: str) -> Dict[str, Any]:
        """Calls LLM to extract variable-related information."""
        prompt = self.get_prompt(
            'extract_variables',
            paper_title=paper_title,
            process_text=process_text,
            markdown_content=markdown_content
        )
        system_prompt = "You are a professional literature information extraction expert. Please strictly follow the requirements to extract information, and only output valid JSON format."
        return self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or {}

    @staticmethod
    def _deduplicate_list(original_list: List[Any]) -> List[Any]:
        """Deduplicates a list, handling unhashable items."""
        try:
            # Fast path for hashable items
            return list(set(original_list))
        except TypeError:
            # Slow path for unhashable items
            new_list = []
            for item in original_list:
                if item not in new_list:
                    new_list.append(item)
            return new_list

    def extract_from_search_results(self, search_results: List[Dict[str, Any]], schema_csv_path: str, num_workers: int = 16) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts structured information for all documents from search results in parallel.
        """
        _, process_text, mechanism_text = self._process_schema_for_prompt(schema_csv_path)

        all_mechanism_results = []
        all_variables_results = {
            "targets_and_metrics": {},
            "optimization_variables": {}
        }

        def _extract_from_paper(paper: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            """
            Worker function to extract mechanism and variables from a single paper.
            """
            title = paper.get("title", "Unknown Title")
            doi = paper.get("doi", "Unknown DOI")
            markdown_content = paper.get("markdown_content", "")
            if not markdown_content:
                return [], {}

            try:
                mechanism_info = self._extract_for_mechanism(markdown_content, process_text, mechanism_text, title)
                variables_info = self._extract_for_variables(markdown_content, process_text, title)
                # 添加论文doi(针对zn试剂场景)
                for item in mechanism_info:
                    item["paper_doi"] = doi
            
                return mechanism_info or [], variables_info or {}
            except Exception as e:
                print(f"Error extracting data from paper '{title}': {e}")
                return [], {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_extract_from_paper, paper) for paper in search_results]
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(search_results), desc="Extracting Structured Data"):
                results.append(future.result())

        for mechanism_info, variables_info in results:
            if mechanism_info:
                all_mechanism_results.extend(mechanism_info)
            if variables_info:
                for key, values in variables_info.get("targets_and_metrics", {}).items():
                    if values is not None:
                        all_variables_results["targets_and_metrics"].setdefault(key, []).extend(values)
                for key, values in variables_info.get("optimization_variables", {}).items():
                    if values is not None:
                        all_variables_results["optimization_variables"].setdefault(key, []).extend(values)

        # 去除重复值
        try:
            for key in all_variables_results["targets_and_metrics"]:
                all_variables_results["targets_and_metrics"][key] = self._deduplicate_list(
                    all_variables_results["targets_and_metrics"][key]
                )
            for key in all_variables_results["optimization_variables"]:
                all_variables_results["optimization_variables"][key] = self._deduplicate_list(
                    all_variables_results["optimization_variables"][key]
                )
        except Exception as e:
            print(f"Error deduplicating variables: {e}")
            print(f"Variables results: {all_variables_results}")
            raise e

        return all_mechanism_results, all_variables_results


    def classify_papers(self, papers: List[Dict[str, Any]], num_workers: int = 16) -> List[Dict[str, Any]]:
        """
        Classifies a list of papers into different categories based on their content concurrently.
        Each paper is a dictionary with the following keys:
        - title: str
        - markdown_content: str
        - doi: str
        - pdf_path: str
        """
        system_prompt = "You are a professional literature classification expert. Please strictly follow the requirements to classify the paper, and only output valid JSON format."

        def _classify_single_paper(paper: Dict[str, Any]) -> Any:
            markdown_content = paper.get("markdown_content", "")
            doi = paper.get("doi", "")
            title = paper.get("title", "")
            prompt = self.get_prompt(
                'classify_papers',
                title=title,
                doi=doi,
                markdown_content=markdown_content
            )
            try:
                result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or {}
                category = result.get("category", "")
                if category:
                    res = paper.copy()
                    res["category"] = category
                    res.pop("markdown_content")
                    return res
            except Exception as e:
                print(f"Error processing paper '{title}': {e}")
            return None

        classified_papers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_classify_single_paper, paper) for paper in papers]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(papers), desc="Classifying Papers"):
                try:
                    res = future.result()
                    if res:
                        classified_papers.append(res)
                except Exception as e:
                    print(f"Error classifying paper: {e}")
        
        return classified_papers
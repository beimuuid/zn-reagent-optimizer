from .config import AppConfig
from knowledge import KnowledgeTree, KnowledgeProcessor
from agents import (
    LiteratureExtractionAgent,
    SchemaGenerationAgent,
    StructuredDataExtractionAgent,
    KnowledgeRefinementAgent,
    VariableExtractionAgent,
    ExplanationAgent,
)
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import traceback
import logging
import pickle
import glob
from utils import extract_title_from_pdf

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

    def parse(self, pdf_path: str) -> str:
        return self._parse_sync(pdf_path)

    def _parse_sync(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text("text") for page in doc)



def step_schema_generation(cfg: AppConfig, agent: SchemaGenerationAgent, search_query: str, output_dir: str, additional_info: Optional[str], logger: logging.Logger) -> Optional[str]:
    """执行 Schema 生成步骤"""
    if cfg.paths.schema_path and os.path.exists(cfg.paths.schema_path):
        logger.info(f"  * [Skip] 使用指定的 Schema 文件: {cfg.paths.schema_path}")
        return cfg.paths.schema_path

    if cfg.paths.experiment_process_path and os.path.exists(cfg.paths.experiment_process_path):
        with open(cfg.paths.experiment_process_path, 'r', encoding='utf-8') as f:
            experiment_process = f.read()
        logger.info(f"  * 实验步骤读取完成: {cfg.paths.experiment_process_path}")
    else:
        logger.warning(f"  * 实验步骤文件不存在: {cfg.paths.experiment_process_path}")
        experiment_process = None

    logger.info(f"\n[Schema] 正在生成 Schema...")
    schema_output_dir = os.path.join(output_dir, "schema")
    schema, csv_path = agent.generate(
        user_query=search_query,
        experiment_process=experiment_process,
        output_dir=schema_output_dir,
        additional_info=additional_info
    )
    if not schema:
        logger.error("  [Error] Schema 生成失败")
        return None
    
    logger.info(f"  * Schema 生成完成, 已保存至: {csv_path}")
    return csv_path
    

def step_literature_search(cfg: AppConfig, agent: LiteratureExtractionAgent, output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行文献检索和内容提取步骤"""
    if cfg.paths.search_results_path and os.path.exists(cfg.paths.search_results_path):
        logger.info(f"  * [Skip] 从缓存加载文献检索结果: {cfg.paths.search_results_path}")
        with open(cfg.paths.search_results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    logger.info(f"\n[Literature] 正在进行文献检索与提取...")
    try:
        search_results = agent.search_and_extract_sync(query=cfg.search_query, max_results_per_query=cfg.agents["literature_extraction"].get("max_results_per_query", 10))
        #search_results = agent.search_for_literature(query=cfg.search_query)
        if search_results:
            logger.info(f"  * 文献检索完成，找到 {len(search_results)} 篇相关文献")
            output_path = os.path.join(output_dir, "search_results.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 检索结果已保存至: {output_path}")
            return search_results
        else:
            logger.warning("  [Warning] 文献检索未返回结果")
            return None
    except Exception as e:
        logger.error(f"  [Error] 文献检索失败: {e}")
        logger.error(traceback.format_exc())
        return None


def step_structured_data_extraction(cfg: AppConfig, agent: StructuredDataExtractionAgent, pdf_path: str, schema_csv_path: str, output_dir: str, logger: logging.Logger) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """执行结构化信息提取步骤"""
    if cfg.paths.mechanism_structured_data_path and os.path.exists(cfg.paths.mechanism_structured_data_path) and \
       cfg.paths.variables_structured_data_path and os.path.exists(cfg.paths.variables_structured_data_path):
        logger.info(f"  * [Skip] 从缓存加载结构化数据: {cfg.paths.mechanism_structured_data_path} 和 {cfg.paths.variables_structured_data_path}")
        with open(cfg.paths.mechanism_structured_data_path, 'r', encoding='utf-8') as f:
            mechanism_data = json.load(f) 
        with open(cfg.paths.variables_structured_data_path, 'r', encoding='utf-8') as f:
            variables_data = json.load(f)
        return mechanism_data, variables_data
    else:
        pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        mock_search_results = []
        data_extraction_agent = agent
        pdf_parser = PDFParser()
        print("✅ Agents 初始化完成。")
        for pdf_file in pdf_files:
            pdf_file_path = os.path.join(pdf_path, pdf_file)
            final_schema_path = schema_csv_path
            try:
                markdown_content = pdf_parser.parse(pdf_file_path)
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
            # if len(mock_search_results) >= 300:
            #     break

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


def step_variables_extraction(cfg: AppConfig, agent: VariableExtractionAgent, variables_structured_data: Dict[str, Any], targets: Dict[str, str], output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行变量提取步骤"""
    if cfg.paths.optimization_variables_path and os.path.exists(cfg.paths.optimization_variables_path):
        logger.info(f"  * [Skip] 从缓存加载变量提取结果: {cfg.paths.optimization_variables_path}")
        with open(cfg.paths.optimization_variables_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not variables_structured_data:
        logger.warning("  [Skip] 因缺少结构化变量数据，跳过变量提取")
        return None

    logger.info(f"\n[Optimization Variables Extraction] 正在提取优化变量...")
    optimization_variables_data = variables_structured_data.get("optimization_variables", [])
    try:
        optimization_variables = agent.extract_variables_from_structured_data(
            variables_structured_data=optimization_variables_data,
            targets=targets
        )
        if optimization_variables:
            logger.info(f"  * 优化变量提取完成")
            output_path = os.path.join(output_dir, "optimization_variables.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(optimization_variables, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 优化变量信息已保存至: {output_path}")
        return optimization_variables
    except Exception as e:
        logger.error(f"  [Error] 优化变量提取失败: {e}")
        logger.error(traceback.format_exc())
        return None

def step_variables_refinement(cfg: AppConfig, agent: KnowledgeRefinementAgent, optimization_variables: Dict[str, Any], additional_context: Optional[str], output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行优化变量精炼步骤"""
    if cfg.paths.refined_optimization_variables_path and os.path.exists(cfg.paths.refined_optimization_variables_path):
        logger.info(f"  * [Skip] 从缓存加载精炼后的优化变量: {cfg.paths.refined_optimization_variables_path}")
        with open(cfg.paths.refined_optimization_variables_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not optimization_variables:
        logger.warning("  [Skip] 因缺少优化变量数据，跳过优化变量精炼")
        return None
        
    logger.info(f"\n[Optimization Variables Refinement] 正在进行优化变量精炼...")
    try:
        refined_data = agent.refine_variables(
            optimization_variables=optimization_variables,
            additional_context=additional_context
        )
        if refined_data:
            logger.info(f"  * 优化变量精炼完成")
            output_path = os.path.join(output_dir, "refined_optimization_variables.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 优化变量精炼结果已保存至: {output_path}")
        return refined_data
    except Exception as e:
        logger.error(f"  [Error] 优化变量精炼失败: {e}")
        logger.error(traceback.format_exc())
        return None

def step_variable_divisions_extraction(cfg: AppConfig, agent: VariableExtractionAgent, mechanism_structured_data: List[Dict[str, Any]], optimization_variables: Dict[str, Any], output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行变量划分提取步骤"""
    if cfg.paths.variable_divisions_path and os.path.exists(cfg.paths.variable_divisions_path):
        logger.info(f"  * [Skip] 从缓存加载变量划分: {cfg.paths.variable_divisions_path}")
        with open(cfg.paths.variable_divisions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 限制机理数据量
    logger.info(f"\n[Variable Divisions Extraction] 正在进行变量划分提取...")
    if len(mechanism_structured_data) >= 30000:
        logger.info(f"  * 机理数据量过大，选择其中30000条数据进行变量划分提取")
        mechanism_structured_data = mechanism_structured_data[:30000]

    try:
        variable_divisions = agent.extract_variable_divisions_from_structured_data(
            mechanism_structured_data=mechanism_structured_data,
            optimization_variables=optimization_variables,
            targets=cfg.optimization.targets,
        )
        if variable_divisions:
            logger.info(f"  * 变量划分提取完成")
            output_path = os.path.join(output_dir, "variable_divisions.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(variable_divisions, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 变量划分已保存至: {output_path}")
        return variable_divisions
    except Exception as e:
        logger.error(f"  [Error] 变量划分提取失败: {e}")
        logger.error(traceback.format_exc())
        return None

def step_variable_divisions_refinement(cfg: AppConfig, agent: KnowledgeRefinementAgent, variable_divisions: List[Dict[str, Any]], explanation_and_feedback_info: Optional[Dict[str, Any]], output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行变量划分精炼步骤"""
    if not variable_divisions:
        logger.warning("  [Skip] 因缺少变量划分数据，跳过变量划分精炼")
        return None
    
    logger.info(f"\n[Variable Divisions Refinement] 正在进行变量划分精炼...")
    try:
        refined_data = agent.refine_variable_divisions(
            variable_divisions=variable_divisions,
            explanation_and_feedback_info=explanation_and_feedback_info
        )
        if refined_data:
            logger.info(f"  * 变量划分精炼完成")
            output_path = os.path.join(output_dir, "refined_variable_divisions.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 变量划分精炼结果已保存至: {output_path}")
        return refined_data
    except Exception as e:
        logger.error(f"  [Error] 变量划分精炼失败: {e}")
        logger.error(traceback.format_exc())
        return None

def step_knowledge_tree_construction(cfg: AppConfig,
                                         variable_divisions: List[Dict[str, Any]],
                                         optimization_variables:Dict[str, Any],
                                         output_dir: str,
                                         logger: logging.Logger):
    """
    构建知识树
    """
    # 加载知识树
    logger.info(f"\n加载知识树...")
    fusion = KnowledgeProcessor(variable_divisions, optimization_variables=optimization_variables)

    tree = KnowledgeTree(
        fusion=fusion,
        optimization_variables=optimization_variables,
        Cp=cfg.knowledge.Cp,
        alpha=cfg.knowledge.alpha,
    )
    exported = tree.export_tree()
    leaf_nodes = [node for node in exported['nodes'] if node['var_name'] is None]
    internal_nodes = [node for node in exported['nodes'] if node['var_name'] is not None]
    logger.info(f"  * 知识树加载完成")
    logger.info(f"  * 知识树节点数: {len(exported['nodes'])}, 叶子节点数: {len(leaf_nodes)}, 内部节点数: {len(internal_nodes)}")
    logger.info(f"  * UCB 参数: Cp={cfg.knowledge.Cp}, alpha={cfg.knowledge.alpha}")

    # 保存知识树到json文件
    tree_path = os.path.join(output_dir, "knowledge_tree.json")
    with open(tree_path, "w", encoding='utf-8') as f:
        json.dump(exported, f, indent=2, ensure_ascii=False)
    logger.info(f"  * 知识树已保存到: {tree_path}")
    return tree
 
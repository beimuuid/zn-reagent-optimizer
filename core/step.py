from optimization import LeafMOBO, HAVE_BOTORCH, make_evaluator
from .config import AppConfig
from utils import aggregate_reward_from_batch, save_checkpoint, load_checkpoint, load_history_csv
from knowledge import KnowledgeTree, KnowledgeProcessor
from agents import (
    LiteratureExtractionAgent,
    SchemaGenerationAgent,
    StructuredDataExtractionAgent,
    KnowledgeRefinementAgent,
    VariableExtractionAgent,
    LiteratureRankingAgent,
    ExplanationAgent,
    ExperimentDecompositionAgent,
    PseudoPointPredictionAgent
)
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import traceback
import logging
import pickle
import glob



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
    
def step_experiment_decomposition(cfg: AppConfig, agent: ExperimentDecompositionAgent, schema_csv_path: str, output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行实验分解步骤"""
    # 读取实验步骤txt文件
    with open(cfg.paths.experiment_process_path, 'r', encoding='utf-8') as f:
        experiment_data = f.read()
    logger.info(f"\n[Experiment Decomposition] 正在进行实验分解...")
    try:
        experiment_decomposition_results = agent.experiment_decomposition(experiment_data, schema_csv_path)
        if experiment_decomposition_results:
            logger.info(f"  * 实验分解完成")
            output_path = os.path.join(output_dir, "experiment_decomposition.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_decomposition_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 实验分解结果已保存至: {output_path}")
            return experiment_decomposition_results
        else:
            logger.warning("  [Warning] 实验分解未返回结果")
            return None
    except Exception as e:
        logger.error(f"  [Error] 实验分解失败: {e}")
        logger.error(traceback.format_exc())
        return None

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

def step_literature_ranking(cfg: AppConfig, agent: LiteratureRankingAgent, search_results: List[Dict[str, Any]], num_results: int, output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行文献排序步骤"""
    if cfg.paths.literature_ranking_path and os.path.exists(cfg.paths.literature_ranking_path):
        logger.info(f"  * [Skip] 从缓存加载文献排序结果: {cfg.paths.literature_ranking_path}")
        with open(cfg.paths.literature_ranking_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    logger.info(f"\n[Literature Ranking] 正在进行文献排序...")
    try:
        literature_ranking_results = agent.literature_ranking(
            search_query=cfg.search_query,
            search_results=search_results
        )
        if literature_ranking_results:
            logger.info(f"  * 文献排序完成")
            output_path = os.path.join(output_dir, "literature_ranking.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(literature_ranking_results, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 文献排序结果已保存至: {output_path}")
            return literature_ranking_results[:num_results]
        else:
            logger.warning("  [Warning] 文献排序未返回结果")
            return None
    except Exception as e:
        logger.error(f"  [Error] 文献排序失败: {e}")
        logger.error(traceback.format_exc())
        return None

def step_structured_data_extraction(cfg: AppConfig, agent: StructuredDataExtractionAgent, search_results: List[Dict[str, Any]], schema_csv_path: str, output_dir: str, logger: logging.Logger) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
    """执行结构化信息提取步骤"""
    if cfg.paths.mechanism_structured_data_path and os.path.exists(cfg.paths.mechanism_structured_data_path) and \
       cfg.paths.variables_structured_data_path and os.path.exists(cfg.paths.variables_structured_data_path):
        logger.info(f"  * [Skip] 从缓存加载结构化数据: {cfg.paths.mechanism_structured_data_path} 和 {cfg.paths.variables_structured_data_path}")
        with open(cfg.paths.mechanism_structured_data_path, 'r', encoding='utf-8') as f:
            mechanism_data = json.load(f)

        #去掉extracted_data字段
        for item in mechanism_data:
            try:
                item.pop("extracted_data")
            except:
                continue
        
        with open(cfg.paths.variables_structured_data_path, 'r', encoding='utf-8') as f:
            variables_data = json.load(f)
        return mechanism_data, variables_data

    if not search_results:
        logger.warning("  [Skip] 因缺少文献检索结果，跳过结构化信息提取")
        return None, None

    logger.info(f"\n[Data Extraction] 正在进行文献结构化信息提取...")
    try:
        mechanism_data, variables_data = agent.extract_from_search_results(
            search_results=search_results,
            schema_csv_path=schema_csv_path
        )
        if mechanism_data and variables_data:
            logger.info(f"  * 结构化信息提取完成")
            mech_output_path = os.path.join(output_dir, "mechanism_structured_data.json")
            vars_output_path = os.path.join(output_dir, "variables_structured_data.json")
            with open(mech_output_path, 'w', encoding='utf-8') as f:
                json.dump(mechanism_data, f, indent=2, ensure_ascii=False)
            with open(vars_output_path, 'w', encoding='utf-8') as f:
                json.dump(variables_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 结构化机理数据已保存至: {mech_output_path}")
            logger.info(f"  * 结构化变量数据已保存至: {vars_output_path}")

        # 去掉extracted_data字段
        for item in mechanism_data:
            try:
                item.pop("extracted_data")
            except:
                continue
        with open(os.path.join(output_dir, "mechanism_structured_data_simple.json"), 'w', encoding='utf-8') as f:
            json.dump(mechanism_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  * 简化结构化机理数据已保存至: {os.path.join(output_dir, 'mechanism_structured_data_simple.json')}")

        return mechanism_data, variables_data
    except Exception as e:
        logger.error(f"  [Error] 结构化信息提取失败: {e}")
        logger.error(traceback.format_exc())
        return None, None


def step_mechanism_refinement(cfg: AppConfig, agent: KnowledgeRefinementAgent, search_query: str, mechanism_data: List[Dict[str, Any]], additional_context: Optional[str], output_dir: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """执行机理知识精炼步骤"""
    if cfg.paths.refined_mechanism_structured_data_path and os.path.exists(cfg.paths.refined_mechanism_structured_data_path):
        logger.info(f"  * [Skip] 从缓存加载精炼后的机理知识: {cfg.paths.refined_mechanism_structured_data_path}")
        with open(cfg.paths.refined_mechanism_structured_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if not mechanism_data:
        logger.warning("  [Skip] 因缺少结构化机理数据，跳过知识精炼")
        return None
        
    logger.info(f"\n[Mechanism Refinement] 正在进行机理知识精炼...")
    try:
        refined_data = agent.refine_mechanism_data(
            mechanism_data=mechanism_data,
            query=search_query,
            additional_context=additional_context
        )
        if refined_data:
            logger.info(f"  * 机理知识精炼完成")
            output_path = os.path.join(output_dir, "refined_mechanism_structured_data.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  * 机理知识精炼结果已保存至: {output_path}")
        return refined_data
    except Exception as e:
        logger.error(f"  [Error] 机理知识精炼失败: {e}")
        logger.error(traceback.format_exc())
        return None

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
 
def step_mobo_optimization(cfg: AppConfig, pseudo_point_prediction_agent: PseudoPointPredictionAgent, explanation_agent: ExplanationAgent, tree: KnowledgeTree, optimization_variables: Dict[str, Any], mechanism_structured_data: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    # 叶内 MOBO
    def _apply_round_updates(
        leaf_candidate_map: List[Dict[str, Any]],
        all_Y_evaluated: List[Dict[str, float]],
        mobo: LeafMOBO,
        tree: KnowledgeTree,
        pareto_XY: List[Tuple[Any, Dict[str, float]]],
        best_aggregated_reward: float,
        best_round: int,
        best_reaction_conditions: Optional[Dict[str, Any]],
        best_targets: Optional[Dict[str, Any]],
        t_round: int,
        cfg: AppConfig,
        logger: logging.Logger
    ) -> Tuple[float, int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """应用一轮评估结束后的所有更新。"""
        
        # 4. 将结果分发回各自的叶子并更新
        logger.info("[回传] 正在将评估结果分发并更新知识...")
        evaluated_idx = 0
        for item in leaf_candidate_map:
            leaf = item['leaf']
            X_leaf = item['candidates']
            num_candidates = len(X_leaf)
            
            Y_leaf = all_Y_evaluated[evaluated_idx : evaluated_idx + num_candidates]
            evaluated_idx += num_candidates

            if not Y_leaf:
                continue
            
            # 更新 MOBO 模型
            mobo.observe_real(X_leaf, Y_leaf)
            
            # 更新总历史数据
            pareto_XY.extend(list(zip(X_leaf, Y_leaf)))
            
            # 计算该叶子的奖励并反向传播
            reward = aggregate_reward_from_batch(Y_leaf, targets=cfg.optimization.targets, ranges=cfg.optimization.ranges, mode=cfg.optimization.mode)
            tree.backprop(leaf, reward)
            logger.info(f"  - 叶子ID {leaf.node_id}: 获得奖励 {reward:.4f}, 已回传")

            # 记录推荐
            for i, (x_dict, y_dict) in enumerate(zip(X_leaf, Y_leaf)):
                point_reward = aggregate_reward_from_batch([y_dict], targets=cfg.optimization.targets, ranges=cfg.optimization.ranges, mode=cfg.optimization.mode)
                if point_reward > best_aggregated_reward:
                    best_aggregated_reward = point_reward
                    best_round = t_round
                    best_reaction_conditions = x_dict
                    best_targets = y_dict
                    logger.info(f"[最佳] 发现新的最佳聚合奖励: {best_aggregated_reward:.4f} (Round {best_round})")
                    logger.info(f"[最佳] 最佳反应条件: {best_reaction_conditions}")
                    logger.info(f"[最佳] 最佳目标值: {best_targets}")

        if cfg.optimization.use_pseudo_point_prediction:
            # 5. 在一轮结束时统一剪枝伪标签
            pseudo_before = len(mobo.X_pseudo)
            mobo.prune_pseudo(sim_th=0.98, drop_ratio=min(0.05 + 0.01 * t_round, 0.20))
            pseudo_after = len(mobo.X_pseudo)
            if pseudo_before != pseudo_after:
                logger.info(f"[剪枝] 伪标签数据: {pseudo_before} -> {pseudo_after} (移除 {pseudo_before - pseudo_after} 条)")

            
        return best_aggregated_reward, best_round, best_reaction_conditions, best_targets


    logger.info(f"\n初始化叶内贝叶斯优化器...")
    mobo = LeafMOBO( 
                    pseudo_point_prediction_agent=pseudo_point_prediction_agent,
                    batch_q=cfg.optimization.batch_q,
                    n_restarts=cfg.optimization.n_restarts,
                    raw_samples=cfg.optimization.raw_samples,
                    targets=cfg.optimization.targets,
                    optimization_variables=optimization_variables,
                    seed=cfg.seed)
    logger.info(f"  * MOBO 初始化完成")
    logger.info(f"  * 批次大小: {cfg.optimization.batch_q}")
    pareto_XY: List[Tuple[Any, Dict[str, float]]] = []
    best_aggregated_reward = -float('inf')
    best_round = 0
    best_reaction_conditions = None
    best_targets = None
    # 中间结果保存
    intermediate_results = []
    start_round = 1
    
    # Checkpoint 设置
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if cfg.optimization.mobo_checkpoint_path and os.path.exists(cfg.optimization.mobo_checkpoint_path):
        logger.info("\n[Checkpoint] 正在尝试从 checkpoint 恢复...")
        checkpoint_state = load_checkpoint(cfg.optimization.mobo_checkpoint_path, logger)
        if checkpoint_state:
            # 恢复状态
            loaded_round = checkpoint_state['round']
            mobo = checkpoint_state['mobo']
            tree = checkpoint_state['tree']
            pareto_XY = checkpoint_state['pareto_XY']
            best_aggregated_reward = checkpoint_state['best_aggregated_reward']
            best_round = checkpoint_state['best_round']
            best_reaction_conditions = checkpoint_state['best_reaction_conditions']
            best_targets = checkpoint_state['best_targets']
            leaf_candidate_map = checkpoint_state['leaf_candidate_map']
            explanation_result = checkpoint_state['explanation_result']
            intermediate_results = checkpoint_state['intermediate_results']
            logger.info(f"[Checkpoint] 已加载第 {loaded_round} 轮的候选点，现在开始评估...")

            # 使用加载的候选点进行评估
            all_X_to_eval_loaded = checkpoint_state['all_X_to_eval']
            evaluator = make_evaluator(eval_mode=cfg.evaluation.mode, excel_path=cfg.optimization.history_csv_path, reactants=cfg.reactants)
            all_Y_evaluated = evaluator.evaluate_batch(all_X_to_eval_loaded, targets=cfg.optimization.targets, ranges=cfg.optimization.ranges)
            logger.info(f"[Checkpoint] * 评估完成")

            # 应用更新
            logger.info(f"[Checkpoint] 正在应用第 {loaded_round} 轮的更新...")
            best_aggregated_reward, best_round, best_reaction_conditions, best_targets = _apply_round_updates(
                leaf_candidate_map=leaf_candidate_map,
                all_Y_evaluated=all_Y_evaluated,
                mobo=mobo, tree=tree, pareto_XY=pareto_XY,
                best_aggregated_reward=best_aggregated_reward, best_round=best_round,
                best_reaction_conditions=best_reaction_conditions, best_targets=best_targets,
                t_round=loaded_round, cfg=cfg, logger=logger
            )
            # 保存中间结果：每一轮探索的叶子节点id，每一轮推荐的候选点，对应目标值和推荐依据（和机理库中的条目相对应，每轮生成），优化树在每一轮更新后的状态
            # 把目标值合并到推荐依据中
            for record in explanation_result:
                record['targets'] = all_Y_evaluated[record['suggested_point_id'] - 1]
            intermediate_results.append({
                'round': loaded_round,
                'leaves': [leaf['leaf'].node_id for leaf in leaf_candidate_map],
                'explanation_result': explanation_result,
                'best_reaction_conditions': best_reaction_conditions,
                'best_targets': best_targets,
            })
            start_round = loaded_round + 1
            logger.info(f"[Checkpoint] 第 {loaded_round} 轮更新完成，下一轮将从 {start_round} 开始。")
        else:
            logger.info("[Checkpoint] 未找到可用的 checkpoint，将从头开始优化。")


    # 回传历史数据
    if start_round == 1:
        if cfg.knowledge.use_history_data and cfg.knowledge.history_csv_path:
            history_data = load_history_csv(cfg.knowledge.history_csv_path, vars=optimization_variables.get("vars", []), targets=cfg.optimization.targets)
            logger.info(f"  * 历史数据加载完成，共 {len(history_data)} 条，开始回传历史数据...")
            pareto_XY.extend(history_data)
            X_history = [item[0] for item in history_data]
            Y_history = [item[1] for item in history_data]
            mobo.observe_real(X_history, Y_history)

            for X, Y in history_data:
                leaf = tree.find_leaf_for_point(X)
                if leaf is None:
                    logger.warning(f"  - [Warning] 历史数据点 {X} 不在搜索空间内，跳过回传")
                    continue
                reward = aggregate_reward_from_batch([Y], targets=cfg.optimization.targets, ranges=cfg.optimization.ranges, mode=cfg.optimization.mode)
                tree.backprop(leaf, reward)
                logger.info(f"  - 历史数据: {X} -> {Y} (奖励: {reward:.4f})，回传到节点ID: {leaf.node_id}")
                logger.info(f"  - 历史数据已回传到知识树和MOBO")  
        else:
            logger.info(f"  * 未使用历史数据，跳过回传历史数据...")
    # 评估
    logger.info(f"\n初始化评估器...")
    evaluator = make_evaluator(eval_mode=cfg.evaluation.mode, excel_path=cfg.paths.excel_path, reactants=cfg.reactants)
    logger.info(f"  * 评估器初始化完成 (模式: {cfg.evaluation.mode})")

    # 优化循环
    logger.info("\n" + "=" * 80)
    logger.info("开始优化循环")
    logger.info("=" * 80)
    while start_round <= cfg.optimization.rounds:
        logger.info(f"\n{'='*80}")
        logger.info(f"Round {start_round}/{cfg.optimization.rounds}")
        logger.info(f"{'='*80}")
         # 生成伪标签数据
        if start_round == 1 and cfg.optimization.use_pseudo_point_prediction and len(mobo.X_pseudo) == 0:
            logger.info(f"[伪标签] 正在生成伪标签数据...")
            mobo._seed_pseudo_with_llm(optimization_variables.get("domains", {}),
                reactants=cfg.reactants,
                mechanism_structured_data=mechanism_structured_data,
                n_samples_per_continuous_var=cfg.optimization.n_samples_per_continuous_var)
            logger.info(f"[伪标签] * 已生成 {len(mobo.X_pseudo)} 个伪标签样本")
        else:
            logger.info(f"[伪标签] 未使用伪点预测，跳过伪点生成")
        # 1. 动态多叶子采样策略
        if cfg.optimization.use_multi_leaf_sampling:
            rounds = cfg.optimization.rounds
            if start_round <= max(1, int(rounds * 0.2)):
                num_leaves = min(3, max(1, int(3 - (start_round - 1) * 5.0 / max(1, int(rounds * 0.2)))))
            else:
                num_leaves = 1
        else:
            num_leaves = 1
        logger.info(f"[叶子选择] 动态策略：选择 {num_leaves} 个叶子进行探索")
        leaves = tree.select_leaf_by_ucb(num_leaves=num_leaves)
        if not leaves:
            logger.warning("[Warning] 未找到叶子节点，跳过本轮")
            continue

        all_X_to_eval = []
        leaf_candidate_map = []
        
        # 2. 为每个选中的叶子生成候选点
        per_leaf_q = max(1, cfg.optimization.batch_q // len(leaves))
        logger.info(f"[建议] 为 {len(leaves)} 个叶子分别生成 {per_leaf_q} 个候选点...")
        for leaf in leaves:
            leaf_box = leaf.box
            logger.info(f"  - 探索叶子ID: {leaf.node_id} (n={leaf.n:.1f}, Q={leaf.Q:.3f}, mean={leaf.mean():.3f})")
            
            X_batch_leaf = mobo.suggest(leaf_box, k=per_leaf_q)
            if not X_batch_leaf:
                logger.warning(f"  - [Warning] 叶子 {leaf.node_id} 未生成候选点")
                continue
            if len(X_batch_leaf) < per_leaf_q:
                logger.warning(f"  - [Warning] 叶子 {leaf.node_id}的变量空间小于候选点数量：{per_leaf_q}，返回{len(X_batch_leaf)}个候选点")

            all_X_to_eval.extend(X_batch_leaf)
            leaf_candidate_map.append({'leaf': leaf, 'candidates': X_batch_leaf})

        if not all_X_to_eval:
            logger.warning("[Warning] 所有叶子均未生成有效候选点，跳过本轮")
            continue

        # 解释
        logger.info(f"[解释] 正在解释候选点...")
        if cfg.optimization.use_explanation:
            explanation_result = explanation_agent.explanation_suggested_points(var_info=optimization_variables, suggested_points=all_X_to_eval, mechanism_data=mechanism_structured_data, targets=cfg.optimization.targets)
            if explanation_result:
                for record in explanation_result:
                    record['leaf_id'] = tree.find_leaf_for_point(all_X_to_eval[int(record['suggested_point_id']) - 1]).node_id
                logger.info(f"[解释] * 解释完成")
                explanation_path = os.path.join(output_dir , "explanation")
                os.makedirs(explanation_path, exist_ok=True)
                with open(os.path.join(explanation_path, f"explanation_result_round_{start_round}.json"), "w", encoding="utf-8") as f:
                    json.dump(explanation_result, f, indent=2, ensure_ascii=False)
                logger.info(f"[解释] * 解释结果已保存至: {os.path.join(explanation_path, f"explanation_result_round_{start_round}.json")}")
            else:
                explanation_result = [
                    {
                        'suggested_point_id': i + 1,
                        'conditions_summary': "No explanation available",
                        'mechanistic_explanation': "No explanation available",
                        'mechanistic_evidence_id': "No explanation available",
                        'leaf_id': None
                    }
                    for i in range(len(all_X_to_eval))
                ]
                logger.warning("[解释] * 解释失败，跳过机理解释步骤")
        else:
            explanation_result = [
                {
                    'suggested_point_id': i + 1,
                    'conditions_summary': "No explanation available",
                    'mechanistic_explanation': "No explanation available",
                    'mechanistic_evidence_id': "No explanation available",
                    'leaf_id': None
                }
                for i in range(len(all_X_to_eval))
            ]
        #保存 checkpoint（评估前）
        state_to_save = {
            'round': start_round,
            'mobo': mobo,
            'tree': tree,
            'pareto_XY': pareto_XY,
            'best_aggregated_reward': best_aggregated_reward,
            'best_round': best_round,
            'best_reaction_conditions': best_reaction_conditions,
            'best_targets': best_targets,
            'leaf_candidate_map': leaf_candidate_map,
            'all_X_to_eval': all_X_to_eval,
            'explanation_result': explanation_result,
            'intermediate_results': intermediate_results
        }
        save_checkpoint(state_to_save, checkpoint_dir, logger)

        # 同时将候选点保存到可读文件，方便外部评估
        candidates_path = os.path.join(checkpoint_dir, f"candidates_round_{start_round}.json")
        with open(candidates_path, 'w', encoding='utf-8') as f:
            json.dump(all_X_to_eval, f, indent=2, ensure_ascii=False)
        logger.info(f"候选点已保存至: {candidates_path}")
        # 如果只生成候选点，则在此处停止
        if hasattr(cfg.optimization, 'generate_candidates_only') and cfg.optimization.generate_candidates_only:
            logger.info(f"已为第 {start_round} 轮生成候选点并保存 checkpoint。根据配置，程序将在此处停止。")
            return intermediate_results

        # 3. 统一评估所有候选点
        logger.info(f"[评估] 正在统一评估所有 {len(all_X_to_eval)} 个候选点...")
        all_Y_evaluated = evaluator.evaluate_batch(all_X_to_eval, targets=cfg.optimization.targets, ranges=cfg.optimization.ranges)
        logger.info(f"[评估] * 评估完成")
        
        # 应用更新
        best_aggregated_reward, best_round, best_reaction_conditions, best_targets = _apply_round_updates(
            leaf_candidate_map=leaf_candidate_map,
            all_Y_evaluated=all_Y_evaluated,
            mobo=mobo,
            tree=tree,
            pareto_XY=pareto_XY,
            best_aggregated_reward=best_aggregated_reward,
            best_round=best_round,
            best_reaction_conditions=best_reaction_conditions,
            best_targets=best_targets,
            t_round=start_round,
                cfg=cfg,
                logger=logger
            )
        # 保存中间结果：每一轮探索的叶子节点id，每一轮推荐的候选点，对应目标值和推荐依据（和机理库中的条目相对应，每轮生成），优化树在每一轮更新后的状态
        # 把目标值合并到推荐依据中
        for record in explanation_result:
            record['targets'] = all_Y_evaluated[record['suggested_point_id'] - 1]
        intermediate_results.append({
            'round': start_round,
            'leaves': [leaf['leaf'].node_id for leaf in leaf_candidate_map],
            'explanation_result': explanation_result,
            'best_reaction_conditions': best_reaction_conditions,
            'best_targets': best_targets,
        })
        
        start_round += 1

    # 优化完成
    logger.info("\n" + "=" * 80)
    logger.info("优化流程完成")
    logger.info("=" * 80)
    
    # 最终统计
    if pareto_XY:
        logger.info(f"[最终统计] 总实验数: {len(pareto_XY)}")
        logger.info(f"[最终统计] 最佳反应条件: {best_reaction_conditions}")
        logger.info(f"[最终统计] 最佳目标值: {best_targets}")
        logger.info(f"[最终统计] 最佳聚合奖励: {best_aggregated_reward:.4f} (Round {best_round})")
    
    timestamped_outdir = output_dir
    logger.info(f"[输出] 所有诊断信息已保存到: {timestamped_outdir}")
    
    logger.info("=" * 80)

    if intermediate_results:
        with open(os.path.join(timestamped_outdir, "intermediate_results.json"), "w", encoding="utf-8") as f:
            json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
        logger.info(f"中间结果已保存至: {os.path.join(timestamped_outdir, "intermediate_results.json")}")
    else:
        logger.warning("中间结果为空，未保存")
    
    return intermediate_results

def step_explanation_and_feedback(cfg: AppConfig, explanation_agent: ExplanationAgent, knowledge_tree: Dict[str, Any], intermediate_results: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    logger.info(f"[解释和反思] 正在解释和反思...")
    explanation_result = explanation_agent.explanation_and_feedbacks(knowledge_tree=knowledge_tree, intermediate_results=intermediate_results)
    if not explanation_result:
        logger.warning("[Warning] 解释和反思步骤失败，跳过解释和反思步骤")
        return None
    explanation_and_feedback_path = os.path.join(output_dir , "explanation_and_feedback")
    os.makedirs(explanation_and_feedback_path, exist_ok=True)
    with open(os.path.join(explanation_and_feedback_path, f"explanation_and_feedback_result.json"), "w", encoding="utf-8") as f:
        json.dump(explanation_result, f, indent=2, ensure_ascii=False)
    logger.info(f"[解释和反思] * 解释和反思结果已保存至: {os.path.join(explanation_and_feedback_path, f"explanation_and_feedback_result.json")}")
    return explanation_result
   
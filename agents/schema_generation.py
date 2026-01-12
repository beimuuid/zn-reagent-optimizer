import json
import re
import os
import time
from typing import Optional
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from .base import BaseKnowledgeAgent
from utils import json_to_csv_schema

class SchemaGenerationAgent(BaseKnowledgeAgent):
    """
    通过多 Agent 协作流程生成知识图谱 Schema
    """
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_architects = cfg.get('num_architects', 5)
        self.enable_clustering = cfg.get('enable_clustering', True)
        self.session_dir = None

    def _sanitize_query_for_filename(self, query: str) -> str:
        """清理 query 字符串，使其适合作为文件名/目录名"""
        sanitized = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', query)
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:50]

    def _step1_decompose(self, user_query: str, experiment_process: Optional[str] = None, additional_info: Optional[str] = None) -> str:
        """Step 1: 问题拆解"""
        print("\n" + "="*60 + "\nStep 1: 问题拆解 Agent\n" + "="*60)
        prompt = self.get_prompt('decomposition', user_query=user_query, experiment_process=experiment_process, additional_info=additional_info)
        system_prompt = "You are a cross-domain knowledge graph construction expert. You excel at using 'first principles' and 'hierarchical thinking' to break down complex user queries into metadata dimensions required for building a database."
        
        result = self._call_llm(prompt, system_prompt=system_prompt)
        
        print(f"拆解结果:\n{result}")
        pathlib.Path(self.session_dir, "step1_decomposition.txt").write_text(result, encoding='utf-8')
        return result

    def _step2_architect_task(self, agent_id: int, user_query: str, step_1_output: str, experiment_process: Optional[str] = None, additional_info: Optional[str] = None) -> Dict[str, Any]:
        """单个架构师 Agent 的任务"""
        print(f"  - 架构师 {agent_id} 开始工作...")
        prompt = self.get_prompt('architect', user_query=user_query, experiment_process=experiment_process, step_1_output=step_1_output, additional_info=additional_info)
        system_prompt = "You are a senior data architect. You are proficient in using 'first principles' and 'hierarchical thinking' to break down complex user queries into metadata dimensions required for building a database."
        
        parsed_json = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        fields = parsed_json.get("fields", []) if parsed_json else []
        print(f"  ✓ 架构师 {agent_id} 完成，生成了 {len(fields)} 个字段")
        
        result = {"agent_id": agent_id, "fields": fields}
        pathlib.Path(self.session_dir, f"step2_architect_{agent_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result

    def _step2_generate_concurrently(self, user_query: str, step_1_output: str, experiment_process: Optional[str] = None, additional_info: Optional[str] = None) -> List[Dict[str, Any]]:
        """Step 2: 并发生成"""
        print("\n" + "="*60 + f"\nStep 2: 并发生成 Agent ({self.num_architects} 个架构师)\n" + "="*60)
        results = []
        with ThreadPoolExecutor(max_workers=self.num_architects) as executor:
            futures = [executor.submit(self._step2_architect_task, i + 1, user_query, experiment_process, step_1_output, additional_info) for i in range(self.num_architects)]
            for future in as_completed(futures):
                results.append(future.result())
        
        results.sort(key=lambda x: x.get("agent_id", 0))
        total_fields = sum(len(r.get('fields', [])) for r in results)
        print(f"\n所有架构师完成，共收集到 {total_fields} 个字段建议")
        return results

    def _step3_aggregate(self, user_query: str, architect_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 3: 归纳与组织"""
        print("\n" + "="*60 + "\nStep 3: 归纳与组织 Agent\n" + "="*60)
        suggestions_text = json.dumps(architect_suggestions, ensure_ascii=False, indent=2)
        prompt = self.get_prompt('aggregator', num_architects=len(architect_suggestions), user_query=user_query, architect_suggestions=suggestions_text)
        system_prompt = "You are a senior schema standardization engineer. You are proficient in using 'first principles' and 'hierarchical thinking' to break down complex user queries into metadata dimensions required for building a database."
        
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        if result:
            print(f"✓ 归纳成功，生成 {len(result)} 个统一字段")
            pathlib.Path(self.session_dir, "step3_aggregation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            print("✗ 归纳失败，无法解析 LLM 返回结果")
        return result or {}

    def _step3_5_expand(self, user_query: str, current_schema: Dict[str, Any], experiment_process: Optional[str] = None, additional_info: Optional[str] = None) -> Dict[str, Any]:
        """Step 3.5: 发散扩展"""
        print("\n" + "="*60 + "\nStep 3.5: 发散扩展 Agent\n" + "="*60)
        schema_text = json.dumps(current_schema, ensure_ascii=False, indent=2)
        prompt = self.get_prompt('expansion', user_query=user_query, current_schema=schema_text, experiment_process=experiment_process, additional_info=additional_info)
        system_prompt = "You are a creative domain expert and data scientist. You excel at lateral thinking and identifying missing dimensions in data schemas."
        
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        if result and "new_fields" in result:
            new_fields = result["new_fields"]
            print(f"✓ 扩展成功，发现了 {len(new_fields)} 个新字段")
            pathlib.Path(self.session_dir, "step3_5_expansion.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            
            # Merge new fields into current_schema
            for field in new_fields:
                field_name = field.get("field_name")
                if field_name and field_name not in current_schema:
                    current_schema[field_name] = {
                        "chinese_name": field.get("chinese_name"),
                        "description": field.get("description"),
                        "extraction_tip": field.get("extraction_tip"),
                        "level": field.get("importance", "Level 2"),
                        "tag": field.get("tag", "[Expansion]")
                    }
        else:
            print("✗ 扩展未发现新字段或解析失败")
        
        return current_schema

    def _step4_reflect(self, user_query: str, current_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: 反思与完善"""
        print("\n" + "="*60 + "\nStep 4: 反思与完善 Agent\n" + "="*60)
        schema_text = json.dumps(current_schema, ensure_ascii=False, indent=2)
        prompt = self.get_prompt('reflection', user_query=user_query, current_schema=schema_text)
        system_prompt = "You are a senior data architect and quality control expert. You are proficient in using 'first principles' and 'hierarchical thinking' to break down complex user queries into metadata dimensions required for building a database."
        
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        if result:
            key_fields_count = sum(1 for f in result.values() if isinstance(f, dict) and f.get("is_key_field"))
            print(f"✓ 反思完成，标记了 {key_fields_count} 个重点字段")
            pathlib.Path(self.session_dir, "step4_reflection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            print("✗ 反思失败，将使用前一阶段的 Schema")
            return current_schema
        return result

    def _step5_cluster(self, final_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: 字段聚类分析"""
        print("\n" + "="*60 + "\nStep 5: 字段聚类分析 Agent\n" + "="*60)
        fields_data = json.dumps(final_schema, ensure_ascii=False, indent=2)
        prompt = self.get_prompt('cluster_analyzer', fields_data=fields_data)
        system_prompt = "You are a senior data architect and knowledge graph expert. You are proficient in using 'first principles' and 'hierarchical thinking' to break down complex user queries into metadata dimensions required for building a database."

        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        if result and "clusters" in result:
            print(f"✓ 聚类成功，生成 {len(result['clusters'])} 个类别")
            pathlib.Path(self.session_dir, "step5_clustering.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            # 将聚类信息合并回主 schema
            field_to_cluster = {
                field["field_name"]: {
                    "cluster_id": cluster.get("cluster_id"),
                    "cluster_name": cluster.get("cluster_name"),
                    "cluster_name_en": cluster.get("cluster_name_en")
                }
                for cluster in result["clusters"]
                for field in cluster.get("fields", [])
            }
            for field_name, cluster_info in field_to_cluster.items():
                if field_name in final_schema:
                    final_schema[field_name].update(cluster_info)
        else:
            print("✗ 聚类失败，将跳过此步骤")
        
        return final_schema

    def generate(self, user_query: str,output_dir: str, experiment_process: Optional[str] = None, additional_info: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """执行完整的五步流程来生成 Schema"""
        self.session_dir = pathlib.Path(output_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        print(f"报告将保存在: {self.session_dir}")

        # Step 1
        step_1_output = self._step1_decompose(user_query, experiment_process, additional_info)
        
        # Step 2
        architect_suggestions = self._step2_generate_concurrently(user_query, experiment_process, step_1_output, additional_info)
        
        # Step 3
        aggregated_schema = self._step3_aggregate(user_query, architect_suggestions)
        if not aggregated_schema: return {}
        
        # Step 3.5
        expanded_schema = self._step3_5_expand(user_query, aggregated_schema, experiment_process, additional_info)
        
        # Step 4
        final_schema = self._step4_reflect(user_query, expanded_schema)
        
        # Step 5 (Optional)
        if self.enable_clustering:
            final_schema = self._step5_cluster(final_schema)
        
        # 保存最终结果
        final_schema_path = self.session_dir / "final_schema.json"
        final_schema_path.write_text(json.dumps(final_schema, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n* 最终 Schema 已保存到: {final_schema_path}")

        # 转换为 CSV
        try:
            csv_path = self.session_dir / "final_schema.csv"
            json_to_csv_schema(final_schema, str(csv_path), str(final_schema_path))
            print(f"* CSV 文件已生成: {csv_path}")
        except Exception as e:
            print(f"[Warning] CSV 转换失败: {e}")
            
        return final_schema, csv_path

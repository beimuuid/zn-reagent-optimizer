import json
from typing import Dict, Any, Optional, List

from .base import BaseKnowledgeAgent

class KnowledgeRefinementAgent(BaseKnowledgeAgent):
    """
    Agent for refining optimization variables and literature fact clusters.
    """
    def refine_mechanism_data(
        self,
        mechanism_data: Dict[str, Any],
        query: str,
        additional_context: Optional[str] = "",
    ) -> Dict[str, Any]:
        """
        Refines mechanism data using external knowledge.
        """
        if not self.enabled:
            return mechanism_data
        
        # Replicate original logic to build `variable_info` string
        mechanism_data_json = json.dumps(mechanism_data, ensure_ascii=False, indent=2)
        
        prompt = self.get_prompt(
            'refine_mechanism_data',
            mechanism_data=mechanism_data_json,
            query=query,
            additional_context=additional_context
        )
        
        system_prompt = "You are a precise scientific knowledge refiner. Return valid JSON only."
        refined_data = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)

        if not refined_data:
            return mechanism_data
        
        return refined_data
    
    def refine_variables(
        self,
        optimization_variables: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refines optimization variables, domains, and types using external knowledge.
        """
        if not self.enabled:
            return optimization_variables
        
        # Replicate original logic to build `variable_info` string
        vars_list = optimization_variables.get("vars", [])
        domains_dict = optimization_variables.get("domains", {})
        var_types_dict = optimization_variables.get("var_types", {})
        var_info_parts = []
        for var in vars_list:
            var_type = var_types_dict.get(var, "discrete")
            domain = domains_dict.get(var)
            if var_type == "continuous":
                if isinstance(domain, (tuple, list)) and len(domain) >= 2:
                    var_info_parts.append(f'变量 "{var}" (连续变量):\n  - 类型: continuous\n  - 定义域: [{domain[0]}, {domain[1]}]')
            else:
                if isinstance(domain, (list, set)):
                    domain_list = sorted(list(domain)) if isinstance(domain, set) else domain
                    var_info_parts.append(f'变量 "{var}" (离散变量):\n  - 类型: discrete\n  - 定义域: {domain_list}')
        variable_info = "\n".join(var_info_parts) if var_info_parts else "无变量信息"

        prompt = self.get_prompt(
            'refine_variables',
            variable_info=variable_info,
            additional_context=additional_context or "无"
        )
        
        system_prompt = "You are a precise optimization expert. Return valid JSON only."
        refined_data = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)

        if not refined_data:
            return optimization_variables

        refined_data.setdefault("vars", optimization_variables.get("vars", []))
        refined_data.setdefault("domains", optimization_variables.get("domains", {}))
        refined_data.setdefault("var_types", optimization_variables.get("var_types", {}))

        return refined_data

    def refine_variable_divisions(
        self,
        variable_divisions: Dict[str, Any],
        explanation_and_feedback_info: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refines variable divisions to cover the entire domain.
        """
        if not self.enabled:
            return variable_divisions

        prompt = self.get_prompt(
            'refine_variable_divisions',
            variable_divisions_info=json.dumps(variable_divisions, ensure_ascii=False, indent=2),
            explanation_and_feedback_info=explanation_and_feedback_info
        )

        system_prompt = "You are a precise scientific knowledge refiner. Return valid JSON only."
        refined_data = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)
        
        if not refined_data:
            return variable_divisions

        refined_data.setdefault("variable_divisions", variable_divisions.get("variable_divisions", []))
        refined_data.setdefault("importance", variable_divisions.get("importance", {}))
        
        return refined_data

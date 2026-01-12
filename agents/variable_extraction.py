import json
import os
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from .base import BaseKnowledgeAgent

class VariableExtractionAgent(BaseKnowledgeAgent):
    """
    Agent for extracting optimization variables and variable divisions from text.
    """

    def extract_variables_from_structured_data(
        self,
        variables_structured_data: Dict[str, Any],
        targets: Dict[str, str] = {"yield":"maximize"}
    ) -> Dict[str, Any]:
        """
        Extracts optimization variables, domains, and types from structured data.
        """
        data_json = json.dumps(variables_structured_data, ensure_ascii=False, indent=2)

        prompt = self.get_prompt(
            'extract_variables',
            variables_structured_data=data_json,
            targets=targets
        )
        
        system_prompt = "You are a precise optimization expert. Return valid JSON only."
        extracted_data = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)

        if not extracted_data:
            return {"vars": [], "domains": {}, "var_types": {}}

        extracted_data.setdefault("vars", [])
        extracted_data.setdefault("domains", {})
        extracted_data.setdefault("var_types", {})
        
        return extracted_data

    def extract_variable_divisions_from_structured_data(
        self,
        mechanism_structured_data: Dict[str, Any],
        optimization_variables: Dict[str, Any],
        targets: Dict[str, str] = {"yield":"maximize"},
    ) -> Dict[str, Any]:
        """
        Extracts variable divisions and variable importance from structured data.
        """
        mechanism_data_json = json.dumps(mechanism_structured_data, ensure_ascii=False, indent=2)

        target_vars = optimization_variables.get("vars", [])
        target_domains = optimization_variables.get("domains", {})
        target_var_types = optimization_variables.get("var_types", {})
        
        vars_info_parts = []
        for var in target_vars:
            var_type = target_var_types.get(var, "discrete")
            domain = target_domains.get(var, [])
            
            if var_type == "continuous":
                if isinstance(domain, (tuple, list)) and len(domain) >= 2:
                    vars_info_parts.append(f'    - "{var}" (连续变量): [{domain[0]}, {domain[1]}]')
            else:
                if isinstance(domain, (list, set)):
                    domain_list = sorted(list(domain)) if isinstance(domain, set) else domain
                    domain_str = ", ".join([str(v) for v in domain_list])
                    vars_info_parts.append(f'    - "{var}" (离散变量): [{domain_str}]')
                else:
                    vars_info_parts.append(f'    - "{var}" (离散变量)')
        
        vars_description = "\n".join(vars_info_parts)

        prompt = self.get_prompt(
            'extract_variable_divisions',
            mechanism_structured_data=mechanism_data_json,
            vars_description=vars_description,
            targets=targets
        )
        
        system_prompt = "You are a precise scientific information extractor. Return valid JSON only."
        extracted_data = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt)

        if not extracted_data:
            return {"variable_divisions": [], "importance": {}}

        extracted_data.setdefault("variable_divisions", [])
        extracted_data.setdefault("importance", {})

        return extracted_data

    
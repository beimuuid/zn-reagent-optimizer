import json
import csv
import pathlib
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from .base import BaseKnowledgeAgent

class ExplanationAgent(BaseKnowledgeAgent):
    def explanation_suggested_points(self, var_info: Dict[str, Any], suggested_points: list[Dict[str, Any]], mechanism_data: list[Dict[str, Any]], targets: list[Dict[str, Any]]) -> Dict[str, Any]:
        
        prompt = self.get_prompt(
            'explanation_suggested_points',
            var_info=var_info,
            suggested_points=suggested_points,
            mechanism_data=mechanism_data,
            targets=targets,
        )
        system_prompt = "You are a professional explanation expert. Please strictly follow the requirements to explain the optimization results, and only output valid JSON format."
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or []
        return result
    
    def explanation_influence_factors(self, reagent_info:list[Dict[str, Any]], factors_info:list[Dict[str, Any]], mechanism_data: list[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.get_prompt(
            'explanation_influence_factors',
            reagent_info=reagent_info,
            factors_info=factors_info,
            mechanism_data=mechanism_data,
        )
        system_prompt = "You are a professional explanation expert. Please strictly follow the requirements to explain the optimization results, and only output valid JSON format."
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or []
        return result
    
    def explanation_influence_catalysts(self, reagent_info:list[Dict[str, Any]], catalyst_info:list[Dict[str, Any]], mechanism_data: list[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.get_prompt(
            'explanation_influence_catalysts',
            reagent_info=reagent_info,
            catalyst_info=catalyst_info,
            mechanism_data=mechanism_data,
        )
        system_prompt = "You are a professional explanation expert. Please strictly follow the requirements to explain the optimization results, and only output valid JSON format."
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or []
        return result

    def explanation_and_feedbacks(self, knowledge_tree: Dict[str, Any], intermediate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.get_prompt(
            'explanation_and_feedback',
            knowledge_tree=knowledge_tree,
            intermediate_results=intermediate_results,
        )
        system_prompt = "You are a professional explanation expert. Please strictly follow the requirements to explain the optimization results, and only output valid JSON format."
        result = self._call_llm_and_parse_json(prompt, system_prompt=system_prompt) or []
        return result
import re
import json
from typing import Dict, Any, Optional
import logging

from utils import call_gemini_api

class BaseKnowledgeAgent:
    """
    所有知识处理 Agent 的基类
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        使用配置字典初始化 Agent
        
        Args:
            cfg: 包含模型参数和 prompts 的配置字典
        """
        self.model_name = cfg.get("model_name", "gemini-2.5-pro")
        self.temperature = cfg.get("temperature", 0.2)
        self.base_url = cfg.get("base_url")
        self.api_key = cfg.get("api_key")
        self.prompts = cfg.get("prompts", {})
        self.max_retries = cfg.get("max_retries", 5)
        self.retry_delay = cfg.get("retry_delay", 10)
        self.enabled = self.base_url and self.api_key and self.model_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def _call_llm(self, 
                  prompt: str, 
                  system_prompt: Optional[str] = None, 
                  thinking_mode: bool = True) -> str:
        """
        调用底层 LLM API
        """
        if not self.enabled:
            raise RuntimeError("Agent is not enabled due to missing configuration (base_url, api_key, or model_name).")
        
        return call_gemini_api(
            prompt=prompt,
            system_prompt=system_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            api_key=self.api_key,
            thinking_mode=thinking_mode,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )

    def _call_llm_and_parse_json(self, 
                                 prompt: str, 
                                 system_prompt: Optional[str] = None,
                                 thinking_mode: bool = True,
                                 ) -> Any:
        """
        调用 LLM 并稳健地解析返回的 JSON 字符串
        """
        raw_response = self._call_llm(prompt, system_prompt, thinking_mode)
        try:
            # 1. 尝试直接解析
            return json.loads(raw_response)
        except json.JSONDecodeError:
            # 2. 尝试从 markdown 代码块中提取 (支持对象 {} 和数组 [])
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass  # 继续尝试下一种方法

            # 3. 尝试提取从第一个 '{' 或 '[' 到最后一个 '}' 或 ']' 的内容
            try:
                start_idx = -1
                end_idx = -1
                
                # 确定是对象还是数组
                first_brace = raw_response.find('{')
                first_bracket = raw_response.find('[')
                
                if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                    # It's an object
                    start_idx = first_brace
                    end_idx = raw_response.rfind('}')
                elif first_bracket != -1:
                    # It's an array
                    start_idx = first_bracket
                    end_idx = raw_response.rfind(']')

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = raw_response[start_idx:end_idx+1]
                    return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed after all attempts. Error: {e}")
                print(f"Original response snippet: {raw_response[:100]}...")
                return None
        
        return None

    def get_prompt(self, key: str, **kwargs) -> str:
        """
        从配置中获取并格式化 prompt 模板
        """
        if key not in self.prompts:
            raise KeyError(f"Prompt key '{key}' not found in agent configuration.")
        return self.prompts[key].format(**kwargs)

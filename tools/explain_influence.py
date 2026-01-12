import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AppConfig
from agents.explanation import ExplanationAgent
from typing import Dict, Any
import json
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    app_config = AppConfig.from_omegaconf(cfg)
    explanation_agent = ExplanationAgent(app_config.agents["explanation"])
    data_path = "/root/optimizer/init_files/zn/categorized_zn/zn_S.json"
    mechanism_data_path = "/root/optimizer/init_files/zn/mechanism_structured_data_simple.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    reagent_info = data['reagents']
    # catalyst_info = data['catalysts']
    factors_info = {
        'catalysts': data['catalysts'],
        'additives': data['additives'],
        'solvents': data['solvents'],
    }

    with open(mechanism_data_path, 'r') as f:
        mechanism_data = json.load(f)
    

    print("Explaining influence catalysts...")
    results = explanation_agent.explanation_influence_factors(reagent_info, factors_info, mechanism_data)
    #results = explanation_agent.explanation_influence_catalysts(reagent_info, catalyst_info, mechanism_data)
    with open(f'{data_path.split(".")[0]}_explanation_1.json', 'w') as f:
        json.dump(results, f, indent=4)
    return
if __name__ == '__main__':
    main()
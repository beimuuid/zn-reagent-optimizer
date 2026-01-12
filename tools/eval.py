import json
import numpy as np
from optimization.evaluator import make_evaluator
eval_mode = "excel"
excel_path = "/root/optimizer/data/OpenScienceLab___Suzuki-Miyaura-coupling-on-nanomole-scale/raw/aap9112_data_file_s1_sci2018nealW.xlsx"
reactants = {"reactant_1":"6-triflatequinoline", "reactant_2":"2b, Boronic Ester"}
targets = {"yield":"maximize"}
ranges = {"yield":[0.0, 100.0]}

evaluator = make_evaluator(eval_mode, excel_path, reactants, targets, ranges)
with open("/root/optimizer/outputs/predictions.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)
predicted_targets = []
real_values = []
results = []
for item in predictions:
    real_value = evaluator._evaluate_one_point(item["reaction_info"], targets, ranges)
    if real_value["yield"] == 0.0:
        continue
    real_values.append(real_value["yield"])
    predicted_targets.append(item["predicted_targets"]["yield"])
    results.append({"reaction_info": item["reaction_info"], "predicted_targets": item["predicted_targets"]["yield"], "real_value": real_value["yield"]})


# 不调包实现一个计算R2值的函数
def calculate_r2(real_values, predicted_targets):
    real_values = np.array(real_values)
    predicted_targets = np.array(predicted_targets)
    print(real_values)
    print(predicted_targets)
    r2 = 1 - (sum((real_values - predicted_targets) ** 2) / sum((real_values - np.mean(real_values)) ** 2))

    return r2
r2 = calculate_r2(real_values, predicted_targets)
print(f"R2值: {r2}")
with open("/root/optimizer/outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
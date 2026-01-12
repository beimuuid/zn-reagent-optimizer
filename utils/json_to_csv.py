"""
将 final_schema.json 转换为 CSV 格式
包含字段名、中文含义、描述、提取提示等信息
"""

import json
import csv
import sys

def detect_file_type(data: dict) -> str:
    """检测文件类型：schema 或 variable_space"""
    if not data:
        return "unknown"
    
    # 取第一个值来判断
    sample_value = list(data.values())[0] if data else {}
    
    # 如果包含 variable_type，则是变量空间文件
    if 'variable_type' in sample_value:
        return "variable_space"
    # 如果包含 level 或 extraction_tip，则是 schema 文件
    elif 'level' in sample_value or 'extraction_tip' in sample_value:
        return "schema"
    else:
        return "unknown"


def json_to_csv(json_file_path, csv_file_path):
    """将 JSON 文件转换为 CSV（支持 schema 和 variable_space）"""
    
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检测文件类型
    file_type = detect_file_type(data)
    
    if file_type == "variable_space":
        return json_to_csv_variable_space(data, csv_file_path, json_file_path)
    elif file_type == "schema":
        return json_to_csv_schema(data, csv_file_path, json_file_path)
    else:
        print(f"⚠️ 无法识别文件类型，尝试按 schema 格式处理...")
        return json_to_csv_schema(data, csv_file_path, json_file_path)


def json_to_csv_schema(data: dict, csv_file_path: str, json_file_path: str):
    """处理 Schema 文件"""
    csv_rows = []
    
    for key, value in data.items():
        # 优先使用生成的中文名称，如果没有则从描述中提取
        chinese_name = value.get('chinese_name', '')
        
        # 如果没有生成的中文名称，则从 description 中提取
        if not chinese_name:
            description = value.get('description', '')
            
            # 提取中文名称的策略
            chinese_name = description
            
            # 先尝试按句号分割
            if '。' in description:
                chinese_name = description.split('。')[0].strip()
            elif '.' in description and description.index('.') < 30:
                chinese_name = description.split('.')[0].strip()
            # 再尝试按逗号分割
            elif '，' in description:
                chinese_name = description.split('，')[0].strip()
            elif ',' in description and description.index(',') < 30:
                chinese_name = description.split(',')[0].strip()
            
            # 如果包含英文括号，提取括号前的中文部分
            if '(' in chinese_name and '（' not in chinese_name:
                chinese_name = chinese_name.split('(')[0].strip()
            
            # 如果中文名称太长，截取前40个字符
            if len(chinese_name) > 40:
                chinese_name = chinese_name[:40] + '...'
        
        # 处理重点字段标记
        is_key_field = value.get('is_key_field', False)
        key_field_mark = '是' if is_key_field else '否'
        
        row = {
            '字段名（英文）': key,
            '字段名（中文）': chinese_name,
            '完整描述': value.get('description', ''),
            '提取提示': value.get('extraction_tip', ''),
            '层级': value.get('level', ''),
            '标签': value.get('tag', ''),
            '重点字段': key_field_mark,
            '聚类ID': value.get('cluster_id', ''),
            '聚类名称': value.get('cluster_name', ''),
            '聚类名称（英文）': value.get('cluster_name_en', ''),
        }
        csv_rows.append(row)
    
    # 写入 CSV 文件
    fieldnames = ['字段名（英文）', '字段名（中文）', '完整描述', '提取提示', '层级', '标签', '重点字段', '聚类ID', '聚类名称', '聚类名称（英文）']
    
    with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✅ Schema 转换完成！")
    print(f"输入文件: {json_file_path}")
    print(f"输出文件: {csv_file_path}")
    print(f"共转换 {len(csv_rows)} 条记录")


def json_to_csv_variable_space(data: dict, csv_file_path: str, json_file_path: str):
    """处理变量空间文件"""
    csv_rows = []
    
    for key, value in data.items():
        # 处理典型值数组，转换为字符串
        typical_values = value.get('typical_values', [])
        if isinstance(typical_values, list):
            typical_values_str = ', '.join(str(v) for v in typical_values)
        else:
            typical_values_str = str(typical_values) if typical_values else ''
        
        # 处理重点变量标记
        is_key_variable = value.get('is_key_variable', False)
        key_variable_mark = '是' if is_key_variable else '否'
        
        row = {
            '变量名（英文）': key,
            '变量名（中文）': value.get('chinese_name', ''),
            '变量类型': value.get('variable_type', ''),
            '取值范围': value.get('value_range', ''),
            '典型值': typical_values_str,
            '重要性': value.get('importance', ''),
            '完整描述': value.get('description', ''),
            '维度': value.get('dimension', ''),
            '标签': value.get('tag', ''),
            '重点变量': key_variable_mark
        }
        csv_rows.append(row)
    
    # 写入 CSV 文件
    fieldnames = ['变量名（英文）', '变量名（中文）', '变量类型', '取值范围', '典型值', '重要性', '完整描述', '维度', '标签', '重点变量']
    
    with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✅ 变量空间转换完成！")
    print(f"输入文件: {json_file_path}")
    print(f"输出文件: {csv_file_path}")
    print(f"共转换 {len(csv_rows)} 条记录")


if __name__ == "__main__":
    import os
    
    # 基础输出目录
    base_output_dir = "/Users/wangweida/Desktop/PJ/BO/output"
    
    # 处理命令行参数
    if len(sys.argv) >= 2:
        # 如果第一个参数是目录名（如 20251119_235001）
        dir_name = sys.argv[1]
        if not dir_name.endswith('.json') and not dir_name.endswith('.csv'):
            # 是目录名，构建完整路径
            default_dir = os.path.join(base_output_dir, dir_name)
        else:
            # 是文件路径，直接使用
            json_file = sys.argv[1]
            csv_file = sys.argv[2] if len(sys.argv) >= 3 else json_file.replace('.json', '.csv')
            json_to_csv(json_file, csv_file)
            exit(0)
    else:
        # 没有参数，使用默认目录
        default_dir = os.path.join(base_output_dir, "20251119_235001")
        print(f"未提供目录参数，使用默认目录: {default_dir}")
        print("用法: python json_to_csv.py <目录名>")
        print("示例: python json_to_csv.py 20251119_235001")
        print()
    
    # 优先查找文件（支持 schema 和 variable_space）
    json_file = None
    csv_file = None
    
    # 尝试查找 schema 文件
    schema_file = os.path.join(default_dir, "final_schema_with_chinese.json")
    if os.path.exists(schema_file):
        json_file = schema_file
        csv_file = os.path.join(default_dir, "final_schema.csv")
    else:
        schema_file = os.path.join(default_dir, "final_schema.json")
        if os.path.exists(schema_file):
            json_file = schema_file
            csv_file = os.path.join(default_dir, "final_schema.csv")
    
    # 尝试查找变量空间文件
    if not json_file:
        var_space_file = os.path.join(default_dir, "final_variable_space.json")
        if os.path.exists(var_space_file):
            json_file = var_space_file
            csv_file = os.path.join(default_dir, "final_variable_space.csv")
    
    # 检查文件是否存在
    if not json_file or not os.path.exists(json_file):
        print(f"❌ 错误: 找不到 JSON 文件")
        print(f"在目录 {default_dir} 中查找以下文件：")
        print(f"  - final_schema.json 或 final_schema_with_chinese.json")
        print(f"  - final_variable_space.json")
        print(f"请检查目录是否存在或文件是否已生成")
        exit(1)
    
    json_to_csv(json_file, csv_file)


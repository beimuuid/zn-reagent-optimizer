import json
import os
import re

# 文件路径
BASE_DIR = '/root/optimizer/init_files/zn'
CATALYST_JSON = os.path.join(BASE_DIR, 'catalyst.json')
PAPERS_JSON = os.path.join(BASE_DIR, 'classified_papers_catalyst.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'categorized_catalysts')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 选出的六种最常见金属类别
CATEGORIES = ['Pd', 'Cu', 'Ni', 'Fe', 'Co', 'Rh']
OUTPUT_DATA = {cat: {'catalysts': [], 'papers': []} for cat in CATEGORIES}

# 金属元素的正则表达式
# 匹配元素符号，且后面不接小写字母（例如匹配 Pd 时避免匹配到可能存在的 Pda 等假象）
PATTERNS = {cat: re.compile(rf'{cat}(?![a-z])') for cat in CATEGORIES}

def classify_catalyst_name(name):
    """根据名称识别催化剂包含的金属种类"""
    categories = []
    for cat, pattern in PATTERNS.items():
        if pattern.search(name):
            categories.append(cat)
    return categories

def main():
    print("开始根据金属元素进行分类...")
    
    # 1. 处理催化剂列表 (catalyst.json)
    print(f"正在读取 {CATALYST_JSON}...")
    try:
        with open(CATALYST_JSON, 'r', encoding='utf-8') as f:
            zn_data = json.load(f)
            catalysts = zn_data.get('catalyst', [])
            
        print(f"发现 {len(catalysts)} 个催化剂。正在分类...")
        count_catalysts = 0
        for c in catalysts:
            cats = classify_catalyst_name(c)
            if cats:
                for cat in cats:
                    OUTPUT_DATA[cat]['catalysts'].append(c)
                count_catalysts += 1
        print(f"成功分类了 {count_catalysts} 个催化剂。")
        
    except Exception as e:
        print(f"处理催化剂时出错: {e}")
        return

    # 2. 处理文献列表 (classified_papers_catalyst.json)
    print(f"正在读取 {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            
        print(f"发现 {len(papers)} 篇文献。正在分类...")
        
        count_papers = 0
        for paper in papers:
            p_cat_str = paper.get('category', '')
            if not p_cat_str or p_cat_str == 'None':
                continue
            
            # 处理可能的逗号分隔，如 "Pd, Ni, Cu"
            p_cats = [c.strip() for c in p_cat_str.split(',')]
            
            added = False
            for p_cat in p_cats:
                if p_cat in CATEGORIES:
                    OUTPUT_DATA[p_cat]['papers'].append(paper)
                    added = True
            
            if added:
                count_papers += 1
                    
        print(f"成功分类了 {count_papers} 篇文献。")
        
    except Exception as e:
        print(f"处理文献时出错: {e}")
        return

    # 3. 写入输出文件
    print(f"正在将输出文件写入 {OUTPUT_DIR}...")
    for cat in CATEGORIES:
        filename = f"catalyst_{cat}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        data = OUTPUT_DATA[cat]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"已创建 {filename}: 包含 {len(data['catalysts'])} 个催化剂, {len(data['papers'])} 篇文献。")

    print("分类完成。")

if __name__ == '__main__':
    main()


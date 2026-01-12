import json
import os
import re

# File paths
BASE_DIR = '/root/optimizer/init_files/zn'
ZN_JSON = os.path.join(BASE_DIR, 'zn.json')
PAPERS_JSON = os.path.join(BASE_DIR, 'classified_papers.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'categorized')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Categories
CATEGORIES = ['Br', 'Cl', 'I', 'F', 'S', 'O']
OUTPUT_DATA = {cat: {'reagents': [], 'papers': []} for cat in CATEGORIES}

# Regex patterns for elements
# Matches Element symbol not followed by lowercase letter (to avoid matching In when looking for I, etc.)
# Br and Cl are already 2 chars, so less ambiguous in this context, but good to be safe.
PATTERNS = {
    'Br': re.compile(r'Br'),
    'Cl': re.compile(r'Cl'),
    'I': re.compile(r'I(?![a-z])'),
    'F': re.compile(r'F(?![a-z])'),
    'S': re.compile(r'S(?![a-z])'),
    'O': re.compile(r'O(?![a-z])')
}

def classify_reagent(smiles):
    # Check all patterns
    categories = []
    
    if PATTERNS['I'].search(smiles): categories.append('I')
    if PATTERNS['Br'].search(smiles): categories.append('Br')
    if PATTERNS['Cl'].search(smiles): categories.append('Cl')
    if PATTERNS['F'].search(smiles): categories.append('F')
    if PATTERNS['S'].search(smiles): categories.append('S')
    if PATTERNS['O'].search(smiles): categories.append('O')
    
    return categories

def main():
    print("Starting classification...")
    
    # 1. Process Reagents
    print(f"Reading {ZN_JSON}...")
    try:
        with open(ZN_JSON, 'r', encoding='utf-8') as f:
            zn_data = json.load(f)
            reagents = zn_data.get('organozinc_reagent', [])
            
        print(f"Found {len(reagents)} reagents. Classifying...")
        count_reagents = 0
        for r in reagents:
            cats = classify_reagent(r)
            if cats:
                for cat in cats:
                    OUTPUT_DATA[cat]['reagents'].append(r)
                count_reagents += 1
            else:
                # Optional: print unclassified for debugging
                # print(f"Unclassified: {r}")
                pass
        print(f"Classified {count_reagents} reagents (note: some may be in multiple categories).")
        
    except Exception as e:
        print(f"Error processing reagents: {e}")
        return

    # 2. Process Papers
    print(f"Reading {PAPERS_JSON}...")
    try:
        with open(PAPERS_JSON, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            
        print(f"Found {len(papers)} papers. Classifying...")
        
        # Mapping from paper category to our keys
        cat_map = {
            'Br-': 'Br',
            'Cl-': 'Cl',
            'I-': 'I',
            'F-': 'F',
            'S2-': 'S',
            'O2-': 'O'
        }
        
        count_papers = 0
        for paper in papers:
            p_cat = paper.get('category')
            if p_cat in cat_map:
                target_cat = cat_map[p_cat]
                OUTPUT_DATA[target_cat]['papers'].append(paper)
                count_papers += 1
            elif p_cat == 'None':
                continue
            else:
                # Try simple matching if map fails (e.g. whitespace)
                p_cat_clean = str(p_cat).strip()
                if p_cat_clean in cat_map:
                    target_cat = cat_map[p_cat_clean]
                    OUTPUT_DATA[target_cat]['papers'].append(paper)
                    count_papers += 1
                    
        print(f"Classified {count_papers} papers.")
        
    except Exception as e:
        print(f"Error processing papers: {e}")
        return

    # 3. Write Outputs
    print(f"Writing output files to {OUTPUT_DIR}...")
    for cat in CATEGORIES:
        filename = f"zn_{cat}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        data = OUTPUT_DATA[cat]
        # Only write if there is data? Or always write?
        # Requirement: "输出六个...json文件" -> implies all 6 even if empty.
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Created {filename}: {len(data['reagents'])} reagents, {len(data['papers'])} papers.")

    print("Done.")

if __name__ == '__main__':
    main()


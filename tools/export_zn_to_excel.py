import os
import json
import pandas as pd

def process_json_to_excel(input_dir, output_file):
    all_data = []
    
    # Get all files ending with _explanation.json in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('_explanation.json')]
    # Sort files to ensure consistent order
    files.sort()
    
    for filename in files:
        # Get anion name, e.g., zn_O_explanation.json -> O
        parts = filename.split('_')
        if len(parts) >= 2:
            anion = parts[1]
        else:
            anion = "Unknown"
        
        # Read corresponding reagents file, e.g., zn_O.json
        reagents_list = []
        reagents_filename = f"zn_{anion}.json"
        reagents_filepath = os.path.join(input_dir, reagents_filename)
        if os.path.exists(reagents_filepath):
            try:
                with open(reagents_filepath, 'r', encoding='utf-8') as rf:
                    reagents_data = json.load(rf)
                    reagents_list = reagents_data.get('reagents', [])
            except Exception as e:
                print(f"Error processing reagents file {reagents_filename}: {e}")
        
        reagents_str = '; '.join(reagents_list)
        
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # If the JSON is a single object instead of a list, wrap it in a list
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    # Create a copy to avoid modifying original if needed
                    record = item.copy()
                    record['Anion'] = anion
                    record['Zinc Reagents'] = reagents_str
                    
                    # Convert DOI list to string for better display in Excel
                    if 'mechanism_data_doi_list' in record and isinstance(record['mechanism_data_doi_list'], list):
                        record['mechanism_data_doi_list'] = '; '.join(record['mechanism_data_doi_list'])
                    
                    all_data.append(record)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
                
    if not all_data:
        print("No data found to write.")
        return

    df = pd.DataFrame(all_data)
    
    # Reorder columns: Anion and Zinc Reagents first
    preferred_order = ['Anion', 'Zinc Reagents']
    cols = preferred_order + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    
    df.to_excel(output_file, index=False)
    print(f"Successfully wrote {len(all_data)} records to {output_file}")

if __name__ == "__main__":
    dir_path = '/root/optimizer/init_files/zn/categorized_zn'
    output_xlsx = '/root/optimizer/init_files/zn/categorized_zn/zn_explanations.xlsx'
    process_json_to_excel(dir_path, output_xlsx)

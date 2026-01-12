import os
import json
import pandas as pd

def process_catalysts_to_excel(input_dir, output_file):
    all_data = []
    
    # Get all files ending with _explanation.json in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('_explanation.json')]
    # Sort files to ensure consistent order
    files.sort()
    
    for filename in files:
        # Get catalyst category, e.g., catalyst_Pd_explanation.json -> Pd
        parts = filename.split('_')
        if len(parts) >= 2:
            cat_category = parts[1]
        else:
            cat_category = "Unknown"
        
        # Read corresponding catalysts list file, e.g., catalyst_Pd.json
        catalysts_list = []
        catalysts_filename = f"catalyst_{cat_category}.json"
        catalysts_filepath = os.path.join(input_dir, catalysts_filename)
        if os.path.exists(catalysts_filepath):
            try:
                with open(catalysts_filepath, 'r', encoding='utf-8') as rf:
                    catalysts_data = json.load(rf)
                    catalysts_list = catalysts_data.get('catalysts', [])
            except Exception as e:
                print(f"Error processing catalysts file {catalysts_filename}: {e}")
        
        catalysts_str = '; '.join(catalysts_list)
        
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # If the JSON is a single object instead of a list, wrap it in a list
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    record = item.copy()
                    # Add category and list columns
                    record['Catalyst Category'] = cat_category
                    record['Catalyst List'] = catalysts_str
                    
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
    
    # Reorder columns: Category and List first
    preferred_order = ['Catalyst Category', 'Catalyst List']
    cols = preferred_order + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    
    df.to_excel(output_file, index=False)
    print(f"Successfully wrote {len(all_data)} records to {output_file}")

if __name__ == "__main__":
    dir_path = '/root/optimizer/init_files/zn/categorized_catalysts'
    output_xlsx = '/root/optimizer/init_files/zn/categorized_catalysts/catalyst_explanations.xlsx'
    process_catalysts_to_excel(dir_path, output_xlsx)


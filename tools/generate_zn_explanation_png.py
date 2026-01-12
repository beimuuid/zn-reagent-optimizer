import json
import math
import textwrap
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager
import matplotlib.patheffects as path_effects

# Register Arial font with Matplotlib
font_path = "/root/.local/share/fonts/Arial.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
else:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

plt.rcParams['font.size'] = 12
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from PIL import Image, ImageDraw, ImageFont

# --- Helper Functions (Adapted from generate_png.py) ---

def _trim_and_fit_to_canvas(img, target_w, target_h, pad_px=14, white_threshold=245):
    """
    裁掉图像四周接近白色的空白，并将内容按比例缩放后居中放入固定大小画布。
    目标：减少 RDKit 分子图常见的“大白边”，让信息更紧凑。
    """
    if img is None:
        return None
    src = img.convert('RGB')
    try:
        import numpy as np
        arr = np.array(src)
        near_white = (arr[:, :, 0] > white_threshold) & (arr[:, :, 1] > white_threshold) & (arr[:, :, 2] > white_threshold)
        ys, xs = np.where(~near_white)
        if ys.size == 0 or xs.size == 0:
            return Image.new('RGB', (target_w, target_h), color='white')
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        cropped = src.crop(bbox)
    except Exception:
        # numpy 不可用或出错时：退化为原图
        cropped = src

    max_w = max(1, target_w - 2 * pad_px)
    max_h = max(1, target_h - 2 * pad_px)
    cw, ch = cropped.size
    if cw <= 0 or ch <= 0:
        return Image.new('RGB', (target_w, target_h), color='white')

    scale = min(max_w / cw, max_h / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new('RGB', (target_w, target_h), color='white')
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas.paste(resized, (x0, y0))
    return canvas

def create_fallback_image(text, width, height, label):
    """
    Creates an image with text for molecules that cannot be parsed.
    """
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw gray border
    draw.rectangle([0, 0, width-1, height-1], outline='#eeeeee', width=2)
    
    font_path = "/root/.local/share/fonts/Arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 12)
    except:
        try:
            print("arial.ttf font not found, trying DejaVuSans.ttf")
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

    # Draw SMILES text
    margin = 10
    char_width = 7 # Estimate
    chars_per_line = (width - 2 * margin) // char_width
    lines = textwrap.wrap(text, width=int(chars_per_line))
    
    # Limit lines
    max_lines = (height - 40) // 15
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "..."

    total_text_height = len(lines) * 15
    current_y = (height - total_text_height) / 2 - 10

    for line in lines:
        text_w = len(line) * char_width
        x = (width - text_w) / 2
        draw.text((x, current_y), line, fill='#e74c3c', font=font)
        current_y += 15

    # Draw Label
    label_text = label
    l_w = len(label_text) * char_width
    draw.text(((width - l_w) / 2, height - 25), label_text, fill='black', font=font)
    
    return img

def add_smiles_to_image(img, text, width, height):
    """
    Adds SMILES text below the molecule image.
    """
    text_height = 40
    new_height = height + text_height
    new_img = Image.new('RGB', (width, new_height), color='white')
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    font_path = "/root/.local/share/fonts/Arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 12)
    except:
        try:
            print("arial.ttf font not found, trying DejaVuSans.ttf")
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
    char_width = 7
    max_chars = width // char_width
    if len(text) > max_chars:
        text = text[:max_chars-3] + "..."
        
    text_w = len(text) * char_width
    x = (width - text_w) / 2
    y = height + (text_height - 15) / 2
    
    draw.text((x, y), text, fill='#555555', font=font)
    return new_img, new_height

def generate_molecule_grid(reagents, img_width_px=300, img_height_px=250, mols_per_row=5):
    """
    Generates a large grid image containing all reagent structures.
    """
    mol_imgs = []
    final_img_height_px = img_height_px
    
    for i, smi in enumerate(reagents):
        label = f"#{i+1}"
        mol = None
        img = None
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            pass 
            
        if mol:
            try:
                AllChem.Compute2DCoords(mol)
                img = Draw.MolToImage(mol, size=(img_width_px, img_height_px), legend=label)
                # 裁白边并放回固定画布，让结构图在格子里更“占满”
                img = _trim_and_fit_to_canvas(img, img_width_px, img_height_px, pad_px=14, white_threshold=245)
            except Exception as e:
                print(f"Error drawing molecule {i+1}: {e}")
                
        if img:
            img_with_text, h = add_smiles_to_image(img, smi, img_width_px, img_height_px)
            mol_imgs.append(img_with_text)
            final_img_height_px = h
        else:
            fallback_h = img_height_px + 40
            mol_imgs.append(create_fallback_image(smi, img_width_px, fallback_h, label))

    # Stitch images
    n_imgs = len(mol_imgs)
    if n_imgs == 0:
        return Image.new('RGB', (img_width_px, final_img_height_px), color='white'), 1
    
    n_rows_mol = math.ceil(n_imgs / mols_per_row)
    grid_width = mols_per_row * img_width_px
    grid_height = n_rows_mol * final_img_height_px
    
    mol_grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    for idx, img in enumerate(mol_imgs):
        row = idx // mols_per_row
        col = idx % mols_per_row
        x = col * img_width_px
        y = row * final_img_height_px
        mol_grid_img.paste(img, (x, y))
        
    return mol_grid_img, n_rows_mol

# --- New Helper Functions for Text Display ---

def draw_text_block(ax, title, text, color_bar='#3498db', fontsize=12):
    """
    Draws a text block with a title and paragraph content.
    """
    ax.axis('off')
    
    # Title - Larger Bold with path effects
    # 注意：不要再绘制装饰线/色条（会与标题产生重叠），并给标题和正文预留足够间距避免重叠
    t = ax.text(0, 0.98, title, fontsize=fontsize+5, fontweight='bold', color='#333333', va='top', fontname='Arial')
    t.set_path_effects([path_effects.Stroke(linewidth=0.6, foreground='#333333'), path_effects.Normal()])
    
    # Content
    # Wrapped width adjusted for the column width
    wrapped_text = "\n".join(textwrap.wrap(text, width=80))
    
    ax.text(0, 0.80, wrapped_text, fontsize=fontsize, color='#555555', va='top', linespacing=1.55, fontname='Arial')

def draw_list_block(ax, title, items, color_bar='#9b59b6', fontsize=12):
    """
    Draws a list of items (e.g. DOIs).
    """
    ax.axis('off')
    
    # Title - Larger Bold with path effects
    t = ax.text(0, 0.98, title, fontsize=fontsize+5, fontweight='bold', color='#333333', va='top', fontname='Arial')
    t.set_path_effects([path_effects.Stroke(linewidth=0.6, foreground='#333333'), path_effects.Normal()])
    
    y_pos = 0.80
    for i, item in enumerate(items):
        ax.text(0.02, y_pos, f"• {item}", fontsize=fontsize, color='#34495e', va='top', fontname='Arial')
        y_pos -= 0.12

def draw_right_big_title(ax, combination_text, fontsize=20):
    """
    右侧大标题（红色）：catalyst/additive/solvent + 组合信息。
    放在右栏顶部，不要贴边，不要跑到整张图右上角边缘。
    """
    ax.axis('off')
    combo = (combination_text or "Unknown Combination").strip()
    title = "catalyst/additive/solvent:\n" + combo
    wrapped = "\n".join(textwrap.wrap(title, width=34))
    t = ax.text(
        0.0, 0.92, wrapped,
        fontsize=fontsize, fontweight='heavy',
        color='#c0392b', va='top', ha='left',
        fontname='Arial'
    )
    t.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='#c0392b'), path_effects.Normal()])

def _estimate_paragraph_height_in(text, wrap_width=80, fontsize=12, linespacing=1.5, title_extra_in=0.55, pad_in=0.20):
    """
    粗略估算一个“标题 + 段落”的高度（英寸），用于让画布随内容自适应，避免固定大高度导致的留白。
    """
    txt = (text or "").strip()
    if not txt:
        n_lines = 1
    else:
        n_lines = max(1, len(textwrap.wrap(txt, width=wrap_width)))
    # 12pt * 1.5 = 18pt，约 0.25 inch/行
    line_in = (fontsize * linespacing) / 72.0
    return title_extra_in + n_lines * line_in + pad_in

def _estimate_list_height_in(items, fontsize=12, base_in=0.70, per_item_in=0.30, pad_in=0.15):
    n = len(items or [])
    return base_in + n * per_item_in + pad_in

# --- Main Generation Function ---

def create_explanation_infographic(reagents, explanation_item, output_file):
    # 1. Prepare Left Side Image (Reagent Grid)
    # 让左侧网格尽量接近“方形”，避免宽而矮导致布局难以紧凑（同时不拉伸、不畸变）
    n_reagents = len(reagents or [])
    cell_w, cell_h = 240, 210
    cell_total_h = cell_h + 40  # add_smiles_to_image 会增加 40px

    def _best_cols(n, min_c=3, max_c=10):
        if n <= 0:
            return 1
        best_c = min(max_c, max(min_c, n))
        best_score = 1e9
        for c in range(min_c, max_c + 1):
            r = math.ceil(n / c)
            # 目标纵横比接近 1：|log(ar)| 越小越好
            ar = (c * cell_w) / (r * cell_total_h)
            score = abs(math.log(max(ar, 1e-6)))
            if score < best_score:
                best_score = score
                best_c = c
        return best_c

    mols_per_row = _best_cols(n_reagents, min_c=3, max_c=10)

    mol_grid_img, n_rows_mol = generate_molecule_grid(reagents, img_width_px=cell_w, img_height_px=cell_h, mols_per_row=mols_per_row)
    mol_w_px, mol_h_px = mol_grid_img.size
    
    # 2. Setup Figure Layout
    
    # 右侧高度：按内容估算；同时把红色组合信息作为右栏大标题（不再放在整张图顶部）
    comm_text = explanation_item.get('reagent_commonality_summary', '')
    act_text = explanation_item.get('activity_analysis', '')
    doi_list = explanation_item.get('mechanism_data_doi_list', []) or []
    combination_text = explanation_item.get('combination', 'Unknown Combination')

    # 大标题高度（经验值，避免挤压正文）
    h_big_title = 1.35
    h_comm = _estimate_paragraph_height_in(comm_text, wrap_width=80, fontsize=12, linespacing=1.55)
    h_act = _estimate_paragraph_height_in(act_text, wrap_width=80, fontsize=12, linespacing=1.55)
    h_doi = _estimate_list_height_in(doi_list, fontsize=12)
    right_content_height = h_big_title + h_comm + h_act + h_doi

    # 关键：左侧保持比例显示，因此根据左栏宽度和图片纵横比推算所需高度，避免“为了文本高度把左图拉伸”
    fig_width = 17.0
    width_ratios = [1.65, 1.0]
    left_w_in = fig_width * (width_ratios[0] / sum(width_ratios))
    left_required_h_in = left_w_in * (mol_h_px / max(1, mol_w_px))

    fig_height = max(left_required_h_in, right_content_height) + 0.35
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#f9f9f9')

    gs_body = gridspec.GridSpec(
        1, 2,
        width_ratios=width_ratios,
        wspace=0.06,
        left=0.03, right=0.99,
        top=0.97, bottom=0.05
    )
    
    # Left Body: Molecules
    ax_mol = fig.add_subplot(gs_body[0])
    # 保持比例，禁止拉伸畸变
    ax_mol.imshow(mol_grid_img, aspect='equal')
    ax_mol.axis('off')
    ax_mol.set_anchor('NW') # Anchor to Top-West (Top-Left)
    
    # Right Body: Info Panels
    gs_right = gridspec.GridSpecFromSubplotSpec(
        4, 1,
        subplot_spec=gs_body[1],
        height_ratios=[h_big_title, h_comm, h_act, h_doi],
        hspace=0.18
    )

    ax_big = fig.add_subplot(gs_right[0])
    draw_right_big_title(ax_big, combination_text, fontsize=20)
                                              
    # 1. Reagent Commonality
    ax_comm = fig.add_subplot(gs_right[1])
    draw_text_block(ax_comm, "Reagent Properties", explanation_item.get('reagent_commonality_summary', ''), color_bar='#f39c12')
    
    # 2. Activity
    ax_act = fig.add_subplot(gs_right[2])
    draw_text_block(ax_act, "Activity Analysis", explanation_item.get('activity_analysis', ''), color_bar='#27ae60')
    
    # 3. DOIs
    ax_doi = fig.add_subplot(gs_right[3])
    draw_list_block(ax_doi, "Key References", explanation_item.get('mechanism_data_doi_list', []), color_bar='#8e44ad')
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Generated explanation image: {output_file}")

def main():
    reagents_path = '/root/optimizer/init_files/zn/categorized_zn/zn_Br.json'
    explanations_path = '/root/optimizer/init_files/zn/categorized_zn/zn_Br_explanation.json'
    output_dir = '/root/optimizer/init_files/zn/categorized_zn/zn_Br_images_new'
    
    try:
        # Load Reagents
        with open(reagents_path, 'r', encoding='utf-8') as f:
            reagents_data = json.load(f)
            reagents = reagents_data.get('reagents', [])
            
        # Load Explanations
        with open(explanations_path, 'r', encoding='utf-8') as f:
            explanations = json.load(f)
            
        print(f"Found {len(reagents)} reagents and {len(explanations)} explanation items.")
        
        # Generate an image for each explanation item
        for i, item in enumerate(explanations):
            # Create a safe filename based on index
            output_filename = f"zn_Br_explanation_{i+1}.png"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Processing item {i+1}/{len(explanations)}: {item.get('combination', 'Unknown')}")
            create_explanation_infographic(reagents, item, output_path)
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


import json
import math
import textwrap
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager
import matplotlib.patheffects as path_effects
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from PIL import Image, ImageDraw, ImageFont

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

# --- Helper Functions ---

def _trim_and_fit_to_canvas(img, target_w, target_h, pad_px=18, white_threshold=245):
    """
    裁掉图像四周接近白色的空白，并将内容按比例缩放后居中放入固定大小画布。
    这样可以显著减少 RDKit 分子图常见的“大白边”。
    """
    if img is None:
        return None

    src = img.convert('RGB')
    # 找非“近白”像素的包围盒
    import numpy as np
    arr = np.array(src)
    near_white = (arr[:, :, 0] > white_threshold) & (arr[:, :, 1] > white_threshold) & (arr[:, :, 2] > white_threshold)
    ys, xs = np.where(~near_white)
    if ys.size == 0 or xs.size == 0:
        # 全白图：直接按目标大小返回
        return Image.new('RGB', (target_w, target_h), color='white')

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cropped = src.crop(bbox)

    # 目标画布内留一点 padding，避免贴边
    max_w = max(1, target_w - 2 * pad_px)
    max_h = max(1, target_h - 2 * pad_px)
    cw, ch = cropped.size
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
    当分子无法解析时，创建一个带有文本的占位图。
    """
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 绘制灰色边框
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

    # 绘制 SMILES 文本
    margin = 10
    char_width = 7 # 估算值
    chars_per_line = (width - 2 * margin) // char_width
    lines = textwrap.wrap(text, width=int(chars_per_line))
    
    # 限制行数
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

    # 绘制标签
    label_text = label
    l_w = len(label_text) * char_width
    draw.text(((width - l_w) / 2, height - 25), label_text, fill='black', font=font)
    
    return img

def add_smiles_to_image(img, text, width, height):
    """
    在分子图像下方添加 SMILES 文本。
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

def generate_single_molecule_img(smi, img_width_px=800, img_height_px=700):
    """
    生成单个分子的结构图。
    参数 img_width_px / img_height_px 用于控制单张分子渲染的宽高（像素）。
    """
    label = "Zinc Reagent"
    mol = None
    img = None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        pass 
        
    if mol:
        try:
            AllChem.Compute2DCoords(mol)
            # Draw.MolToImage 的 size 参数直接决定分子结构图的输出尺寸
            img = Draw.MolToImage(mol, size=(img_width_px, img_height_px), legend=label)
            # 裁白边并按固定画布居中，减少空白区域
            img = _trim_and_fit_to_canvas(img, img_width_px, img_height_px, pad_px=18, white_threshold=245)
        except Exception as e:
            print(f"Error drawing molecule: {e}")
            
    if img:
        # add_smiles_to_image 会在底部增加文字区域，高度在函数中固定为 40px
        # 返回的新图像尺寸为 (img_width_px, img_height_px + 40)
        img_with_text, h = add_smiles_to_image(img, smi, img_width_px, img_height_px)
        return img_with_text
    else:
        fallback_h = img_height_px + 40
        return create_fallback_image(smi, img_width_px, fallback_h, label)

def draw_text_block(ax, title, text, color_bar='#3498db', fontsize=12, wrap_width=92):
    """
    绘制带有标题和段落内容的文本块。
    """
    ax.axis('off')
    
    # Title - Larger Bold with path effects
    t = ax.text(0, 1.0, title, fontsize=fontsize+6, fontweight='bold', color='#333333', va='top', fontname='Arial')
    t.set_path_effects([path_effects.Stroke(linewidth=0.6, foreground='#333333'), path_effects.Normal()])
    
    # 装饰条（原先会与标题区域发生重叠；按需求移除）
    
    # 内容
    # 自动换行
    # 右侧区域较宽时，70 字符会导致右侧空白较大；提高 wrap_width 可更充分占用宽度
    wrapped_text = "\n".join(textwrap.wrap(text, width=wrap_width))
    
    ax.text(0, 0.84, wrapped_text, fontsize=fontsize, color='#444444', va='top', linespacing=1.7, fontname='Arial')

def draw_list_block(ax, title, items, color_bar='#9b59b6', fontsize=12):
    """
    绘制项目列表（如 DOI）。
    """
    ax.axis('off')
    
    # Title - Larger Bold with path effects
    t = ax.text(0, 1.0, title, fontsize=fontsize+6, fontweight='bold', color='#333333', va='top', fontname='Arial')
    t.set_path_effects([path_effects.Stroke(linewidth=0.6, foreground='#333333'), path_effects.Normal()])
    
    # 装饰条（原先会与标题区域发生重叠；按需求移除）
    
    y_pos = 0.88
    if not items:
        ax.text(0.02, y_pos, "No references available.", fontsize=fontsize, color='#777777', va='top', style='italic', fontname='Arial')
        return

    for item in items:
        ax.text(0.02, y_pos, f"• {item}", fontsize=fontsize, color='#34495e', va='top', fontname='Arial')
        y_pos -= 0.072

def create_item_infographic(item, output_file):
    """
    为单个条目创建信息图。
    """
    smi = item.get('reagent', 'Unknown SMILES')
    
    # 1. 生成分子图（这里指定分子图区域大小：900 x 720 像素，底部文字再加 40px）
    mol_img = generate_single_molecule_img(smi, img_width_px=900, img_height_px=720)
    
    # 2. 设置布局
    # fig_width / fig_height 控制整张信息图的尺寸（单位：英寸），进一步收缩画布以减少边缘空白
    # 画布略收窄，使文字区域更“充满”，减少右侧大块空白
    fig_width = 14.8
    fig_height = 10.2
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#fdfdfd')
    
    # 主网格：1行2列
    # GridSpec 的宽高比例决定左右区域在整张图中的占比；边距(top/bottom/left/right)也会影响可用绘图空间
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.05, 1.15],
        wspace=0.02,
        left=0.02, right=0.992,
        top=0.94, bottom=0.055
    )
    
    # --- 左侧：试剂结构 ---
    ax_mol = fig.add_subplot(gs[0])
    ax_mol.imshow(mol_img)
    ax_mol.axis('off')
    # ax_mol.set_title("Reagent Structure", fontsize=24, fontweight='bold', pad=30, color='#2c3e50')
    t_mol = ax_mol.text(0.5, 1.02, "Zinc Reagent Structure", transform=ax_mol.transAxes, 
                fontsize=24, fontweight='bold', ha='center', color='#2c3e50', fontname='Arial')
    t_mol.set_path_effects([path_effects.Stroke(linewidth=1.0, foreground='#2c3e50'), path_effects.Normal()])

    # --- 右侧：信息板块 ---
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs[1],
        height_ratios=[1.1, 1.1, 0.8],
        hspace=0.08
    )
    
    # 1. Catalyst Commonality Summary
    ax_comm = fig.add_subplot(gs_right[0])
    draw_text_block(ax_comm, "Catalyst Commonality Summary", item.get('catalyst_commonality_summary', ''), color_bar='#f39c12', wrap_width=98)
    
    # 2. Activity Analysis
    ax_act = fig.add_subplot(gs_right[1])
    draw_text_block(ax_act, "Activity Analysis", item.get('activity_analysis', ''), color_bar='#27ae60', wrap_width=98)
    
    # 3. Key References
    ax_doi = fig.add_subplot(gs_right[2])
    draw_list_block(ax_doi, "Key References (DOI)", item.get('mechanism_data_doi_list', []), color_bar='#8e44ad')
    
    # 添加一个底部页脚
    fig.text(0.5, 0.018, "Automated Catalyst Performance Analysis Infographic", ha='center', fontsize=10, color='#bdc3c7', style='italic')

    # 保存
    # 保存时的 dpi 影响输出文件的像素尺寸：实际像素约为 figsize * dpi
    # bbox_inches / pad_inches 会微调留白，间接影响最终像素大小
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Successfully generated: {output_file}")

def main():
    # 输入文件路径
    json_path = '/root/optimizer/init_files/zn/categorized_catalysts/catalyst_Co_explanation.json'
    # 输出目录
    output_dir = '/root/optimizer/init_files/zn/categorized_catalysts/catalyst_Co_images_new'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Found {len(data)} entries. Starting generation...")
        
        for i, item in enumerate(data):
            # 获取试剂的简短描述或使用索引
            output_filename = f"catalyst_Co_explanation_{i+1}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"[{i+1}/{len(data)}] Processing: {item.get('reagent', 'Unknown')[:30]}...")
            create_item_infographic(item, output_path)
            
        print("\nAll images have been generated successfully.")
        print(f"Output folder: {output_dir}")
        
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


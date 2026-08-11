"""
generate_report_pdf.py - 纯 Python 生成灾情评估报告 PDF
包含：染色底图 + 统计数据 + 图例
A3 横向布局
"""

"""
generate_report_pdf.py - 纯 Python 生成灾情评估报告 PDF (A3 横向)
"""

"""
generate_report_pdf.py - 纯 Python 生成灾情评估报告 PDF (A3 横向)
"""

"""
generate_report_pdf.py - 纯 Python 生成灾情评估报告 PDF (A3 横向)
不依赖外部 stats.json 和 legend.txt，内部自动生成
"""
import os
import sys
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def calculate_damage_area(buildings, damage_data, transform, shape, damage_tif):
    """
    计算损毁面积：用 UTM 投影计算每个损毁像素的实际面积
    """
    # 1. 计算单个像素面积（用 UTM 投影）
    with rasterio.open(damage_tif) as src:
        # 获取影像中心点经纬度
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        
        # 计算 UTM 分带
        utm_zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            epsg_code = 32600 + utm_zone
        else:
            epsg_code = 32700 + utm_zone
        
        # 创建 UTM 投影
        import pyproj
        from pyproj import Transformer
        
        # 创建经纬度 -> UTM 转换器
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}")
        
        # 获取影像四个角的 UTM 坐标，计算像素分辨率
        x1, y1 = transformer.transform(bounds.bottom, bounds.left)
        x2, y2 = transformer.transform(bounds.bottom, bounds.right)
        x3, y3 = transformer.transform(bounds.top, bounds.left)
        
        # 像素宽度和高度（米）
        pixel_width = abs(x2 - x1) / src.width
        pixel_height = abs(y3 - y1) / src.height
        pixel_area = pixel_width * pixel_height
        
        print(f"  Pixel size: {pixel_width:.4f} x {pixel_height:.4f} m")
        print(f"  Pixel area: {pixel_area:.4f} m2")
    
    total_damage_area = 0
    level_areas = {2: 0, 3: 0, 4: 0}
    
    for idx, row in buildings.iterrows():
        level = row['level']
        if level in [2, 3, 4]:
            try:
                mask = features.geometry_mask(
                    [row.geometry],
                    out_shape=shape,
                    transform=transform,
                    invert=True
                )
                # 统计损毁像素数
                damage_pixels = np.sum(mask)
                # 计算面积
                area = damage_pixels * pixel_area
                total_damage_area += area
                level_areas[level] += area
            except:
                pass
    
    return total_damage_area, level_areas


def generate_report_pdf(base_tif, building_shp, damage_tif, output_pdf):
    """生成灾情评估报告 PDF (A3 横向)"""
    
    print("Loading data...")
    
    # 损毁等级颜色
    colors = {2: (255, 204, 0), 3: (255, 136, 0), 4: (204, 0, 0)}
    labels = {0: 'No Building', 1: 'No Damage', 2: 'Minor', 3: 'Moderate', 4: 'Severe'}
    
    # 读取底图
    with rasterio.open(base_tif) as src:
        if src.count >= 3:
            img = src.read([1, 2, 3])
        else:
            img = src.read(1)
            img = np.stack([img, img, img], axis=0)
        if img.max() > 255:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        transform = src.transform
        shape = (src.height, src.width)
    
    img = np.transpose(img, (1, 2, 0))
    
    # 读取建筑
    buildings = gpd.read_file(building_shp)
    print(f"Buildings: {len(buildings)}")
    
    # 对齐CRS
    with rasterio.open(damage_tif) as src:
        damage_crs = src.crs
    if buildings.crs != damage_crs:
        buildings = buildings.to_crs(damage_crs)
    
    # 分配损毁等级
    buildings['level'] = 0
    with rasterio.open(damage_tif) as src:
        damage = src.read(1)
        for idx, row in buildings.iterrows():
            try:
                mask = features.geometry_mask([row.geometry], shape, transform, invert=True)
                vals = damage[mask]
                if len(vals) > 0:
                    buildings.loc[idx, 'level'] = int(vals.max())
            except:
                pass
    
    # ========== 统计 ==========
    total = len(buildings)
    damaged = len(buildings[buildings['level'] >= 2])
    counts = {2: len(buildings[buildings['level'] == 2]),
              3: len(buildings[buildings['level'] == 3]),
              4: len(buildings[buildings['level'] == 4])}
    
    # ========== 计算损毁面积（使用 UTM 投影） ==========
    total_damage_area, level_areas = calculate_damage_area(
        buildings, damage, transform, shape, damage_tif
    )
    
    print(f"  Total damage area: {total_damage_area:.2f} m2")
    print(f"  Level 2: {level_areas[2]:.2f} m2")
    print(f"  Level 3: {level_areas[3]:.2f} m2")
    print(f"  Level 4: {level_areas[4]:.2f} m2")
    
    # 叠加颜色
    overlay = img.copy()
    for idx, row in buildings.iterrows():
        if row['level'] in [2, 3, 4]:
            try:
                mask = features.geometry_mask([row.geometry], shape, transform, invert=True)
                overlay[mask] = colors[row['level']]
            except:
                pass
    
    # 生成PDF
    print("Generating PDF...")
    fig = plt.figure(figsize=(16.53, 11.69), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.7, 0.3])
    
    # 地图
    ax = plt.subplot(gs[0, 0])
    ax.imshow(overlay)
    ax.set_title('Damage Assessment Map', fontsize=20, fontweight='bold')
    ax.set_axis_off()
    
    # 信息面板
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    
    y = 0.92
    ax2.text(0.05, y, 'DAMAGE REPORT', fontsize=22, fontweight='bold')
    y -= 0.08
    
    ax2.text(0.05, y, 'LEGEND', fontsize=16, fontweight='bold')
    y -= 0.06
    for level in [4, 3, 2]:
        c = colors[level]
        ax2.text(0.05, y, '###', fontsize=20, color=f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}')
        ax2.text(0.2, y, labels[level], fontsize=13)
        y -= 0.05
    ax2.text(0.05, y, '---', fontsize=20, color='gray')
    ax2.text(0.2, y, 'No Damage', fontsize=13)
    y -= 0.08
    
    ax2.text(0.05, y, 'STATISTICS', fontsize=16, fontweight='bold')
    y -= 0.06
    ax2.text(0.05, y, f'Total Buildings: {total}', fontsize=13)
    y -= 0.05
    ax2.text(0.05, y, f'Damaged: {damaged} ({damaged/total*100:.1f}%)', fontsize=13)
    y -= 0.05
    for level in [2, 3, 4]:
        if counts[level] > 0:
            ax2.text(0.05, y, f'  {labels[level]}: {counts[level]} ({counts[level]/total*100:.1f}%)', fontsize=12)
            y -= 0.04
    
    # ========== 损毁面积 ==========
    y = 0.20
    ax2.text(0.05, y, 'DAMAGE AREA', fontsize=14, fontweight='bold')
    y -= 0.05
    
    if total_damage_area > 0:
        ax2.text(0.05, y, f'Total: {total_damage_area:.2f} m2', fontsize=12)
        y -= 0.04
        for level in [2, 3, 4]:
            if level_areas[level] > 0:
                ax2.text(0.05, y, f'  {labels[level]}: {level_areas[level]:.2f} m2', fontsize=11)
                y -= 0.04
    else:
        ax2.text(0.05, y, 'No damage area detected', fontsize=12, color='gray')
    
    plt.tight_layout()
    
    # 保存
    output_pdf = os.path.abspath(output_pdf)
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PDF saved: {output_pdf}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--building", required=True)
    p.add_argument("--damage", required=True)
    p.add_argument("--outpdf", required=True)
    args = p.parse_args()
    generate_report_pdf(args.base, args.building, args.damage, args.outpdf)
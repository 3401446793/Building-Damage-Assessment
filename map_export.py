import os
import sys
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import mapping
import warnings
import json
warnings.filterwarnings('ignore')


def generate_overlay_with_base(base_tif, building_shp, damage_tif, output_tif):
    """
    生成底图 + 建筑损毁着色的叠加 TIF（RGB）
    只有损毁等级 2、3、4 才染色，0 和 1 显示底图
    """
    print("Reading files...")
    
    # 1. 读取底图
    with rasterio.open(base_tif) as src:
        if src.count >= 3:
            base = src.read([1, 2, 3])
        else:
            base = src.read(1)
            base = np.stack([base, base, base], axis=0)
        
        meta = src.meta.copy()
        meta.update(count=3, dtype=np.uint8, compress='lzw')
        if 'nodata' in meta:
            if meta['nodata'] < 0:
                del meta['nodata']
            else:
                meta['nodata'] = 0
        
        transform = src.transform
        out_shape = (src.height, src.width)
    
    # 归一化到 0-255
    if base.max() > 255:
        base = (base / base.max() * 255).astype(np.uint8)
    else:
        base = base.astype(np.uint8)
    
    # 2. 读取建筑
    # 读取建筑
    buildings = gpd.read_file(building_shp)
    print(f"  Buildings: {len(buildings)}")
    print(f"  Building CRS: {buildings.crs}")

    # ✅ 如果 building.shp 没有 CRS，设置一个
    if buildings.crs is None:
        print("  Warning: building.shp has no CRS, setting to EPSG:4326")
        buildings = buildings.set_crs("EPSG:4326", allow_override=True)

    # 对齐 CRS
    with rasterio.open(damage_tif) as src:
        damage_crs = src.crs
    if buildings.crs != damage_crs:
        buildings = buildings.to_crs(damage_crs)
    
    # 3. 损毁等级颜色 (RGB) - 只对 2、3、4 染色
    damage_colors = {
        2: (255, 204, 0),     # 黄色 - 轻微
        3: (255, 136, 0),     # 橙色 - 中度
        4: (204, 0, 0)        # 红色 - 重度
    }
    
    # 4. 为每个建筑分配损毁等级
    print("  Assigning damage levels...")
    buildings['damage_level'] = 0
    
    with rasterio.open(damage_tif) as src:
        damage_data = src.read(1)
        
        for idx, building in buildings.iterrows():
            try:
                mask_geom = features.geometry_mask(
                    [building.geometry],
                    out_shape=out_shape,
                    transform=transform,
                    invert=True
                )
                values = damage_data[mask_geom]
                if len(values) > 0:
                    damage_level = int(values.max())
                    buildings.loc[idx, 'damage_level'] = damage_level
            except:
                continue
    
    # 5. 创建 RGB 输出数组（先复制底图）
    output_rgb = np.zeros((3, out_shape[0], out_shape[1]), dtype=np.uint8)
    output_rgb[:] = base[:]
    
    # 6. 只烧录损毁建筑（2、3、4 级）
    print("  Burning damaged building colors...")
    
    for idx, building in buildings.iterrows():
        level = building['damage_level']
        if level in [2, 3, 4]:
            try:
                mask_geom = features.geometry_mask(
                    [building.geometry],
                    out_shape=out_shape,
                    transform=transform,
                    invert=True
                )
                color = damage_colors[level]
                output_rgb[0, mask_geom] = color[0]
                output_rgb[1, mask_geom] = color[1]
                output_rgb[2, mask_geom] = color[2]
            except:
                continue
    
    # 7. 保存
    with rasterio.open(output_tif, 'w', **meta) as dst:
        dst.write(output_rgb)
    
    print(f"  Overlay RGB TIF saved: {output_tif}")
    return output_rgb, buildings


def calculate_area_statistics(buildings):
    """
    计算面积统计（平方米）
    需要将建筑投影到 UTM 坐标系
    """
    # 复制建筑数据并投影到 UTM
    buildings_proj = buildings.copy()
    
    # 如果当前是经纬度，投影到 UTM
    if buildings_proj.crs and buildings_proj.crs.is_geographic:
        # 计算中心点经纬度
        center_lon = buildings_proj.geometry.centroid.x.mean()
        center_lat = buildings_proj.geometry.centroid.y.mean()
        utm_zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            utm_crs = f"EPSG:{32600 + utm_zone}"
        else:
            utm_crs = f"EPSG:{32700 + utm_zone}"
        buildings_proj = buildings_proj.to_crs(utm_crs)
        print(f"  Projected to UTM: {utm_crs}")
    
    # 计算每个建筑的面积（平方米）
    buildings_proj['area_m2'] = buildings_proj.geometry.area
    
    # 按损毁等级分组统计面积
    area_stats = {}
    
    # 只统计损毁建筑（2、3、4 级）
    for level in [2, 3, 4]:
        subset = buildings_proj[buildings_proj['damage_level'] == level]
        if not subset.empty:
            area_stats[level] = {
                'count': len(subset),
                'total_area': subset['area_m2'].sum(),
                'min_area': subset['area_m2'].min(),
                'max_area': subset['area_m2'].max(),
                'avg_area': subset['area_m2'].mean()
            }
        else:
            area_stats[level] = {
                'count': 0,
                'total_area': 0,
                'min_area': 0,
                'max_area': 0,
                'avg_area': 0
            }
    
    return area_stats


def generate_statistics(buildings):
    """只统计损毁建筑（2、3、4 级）的数量"""
    stats = {
        'total_buildings': len(buildings),
        'damaged_buildings': len(buildings[buildings['damage_level'] >= 2]),
        'damage_levels': {}
    }
    
    for level in [2, 3, 4]:
        count = len(buildings[buildings['damage_level'] == level])
        stats['damage_levels'][level] = {
            'count': count,
            'percentage': count / len(buildings) * 100 if len(buildings) > 0 else 0
        }
    
    return stats


def generate_legend():
    """只显示损毁图例"""
    return {
        2: {'label': '轻微损毁', 'color': '#ffcc00'},
        3: {'label': '中度损毁', 'color': '#ff8800'},
        4: {'label': '重度损毁', 'color': '#cc0000'}
    }

def generate_png(output_tif, output_png):
    """
    从 RGB TIF 生成 PNG 预览图
    """
    try:
        print("  Generating PNG preview...")
        
        import matplotlib.pyplot as plt
        
        with rasterio.open(output_tif) as src:
            rgb = src.read([1, 2, 3])
            # 转置为 (H, W, C)
            rgb = np.transpose(rgb, (1, 2, 0))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(rgb)
        ax.set_title('Damage Overlay Preview', fontsize=14)
        ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  PNG saved: {output_png}")
        return output_png
        
    except Exception as e:
        print(f"  WARNING: PNG generation failed: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="原始遥感影像 (RGB)")
    parser.add_argument("--building", required=True, help="建筑矢量 SHP")
    parser.add_argument("--damage", required=True, help="损毁分级栅格 TIF")
    parser.add_argument("--outtif", required=True, help="输出叠加 RGB TIF")
    parser.add_argument("--outpng", required=False, help="输出 PNG 预览图")
    parser.add_argument("--outstats", required=False, help="输出统计JSON")
    parser.add_argument("--outlegend", required=False, help="输出图例文本")
    args = parser.parse_args()
    
    # 1. 生成叠加 TIF
    print("\n" + "=" * 60)
    print("STEP 1: Generating overlay with base image")
    print("=" * 60)
    
    result, buildings = generate_overlay_with_base(
        args.base, args.building, args.damage, args.outtif
    )
    
    # 2. 生成 PNG 预览（如果指定了 --outpng）
    if args.outpng:
        print("\n" + "=" * 60)
        print("STEP 2: Generating PNG preview")
        print("=" * 60)
        generate_png(args.outtif, args.outpng)
    
    # 2. 数量统计
    print("\n" + "=" * 60)
    print("STEP 2: Count Statistics")
    print("=" * 60)
    
    stats = generate_statistics(buildings)
    legend = generate_legend()
    
    print("\nLEGEND (Only damaged buildings):")
    for level, info in legend.items():
        print(f"  {level}: {info['label']} ({info['color']})")
    
    total = stats['total_buildings']
    damaged = stats['damaged_buildings']
    print(f"\nTotal buildings: {total}")
    print(f"Damaged buildings: {damaged} ({damaged/total*100:.1f}%)")
    print()
    for level, info in stats['damage_levels'].items():
        label = legend[level]['label']
        print(f"  {label}: {info['count']} ({info['percentage']:.1f}%)")
    
    # 3. 面积统计
    print("\n" + "=" * 60)
    print("STEP 3: Area Statistics (Damaged buildings only)")
    print("=" * 60)
    
    area_stats = calculate_area_statistics(buildings)
    
    print("\nArea Statistics (平方米):")
    total_area = 0
    for level in [2, 3, 4]:
        info = area_stats[level]
        label = legend[level]['label']
        if info['count'] > 0:
            print(f"\n  {label}:")
            print(f"    数量: {info['count']}")
            print(f"    总面积: {info['total_area']:.2f} m2")
            print(f"    最小面积: {info['min_area']:.2f} m2")
            print(f"    最大面积: {info['max_area']:.2f} m2")
            print(f"    平均面积: {info['avg_area']:.2f} m2")
            total_area += info['total_area']
        else:
            print(f"\n  {label}: 0 个建筑")
    
    print(f"\n  损毁建筑总面积: {total_area:.2f} m2")
    
    # 4. 保存
    output_data = {
        'statistics': stats,
        'area_statistics': area_stats,
        'legend': legend
    }
    
    if args.outstats:
        with open(args.outstats, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Statistics saved: {args.outstats}")
    
    if args.outlegend:
        with open(args.outlegend, 'w', encoding='utf-8') as f:
            f.write("LEGEND (Only damaged buildings)\n")
            f.write("=" * 40 + "\n")
            for level, info in legend.items():
                f.write(f"  {level}: {info['label']} ({info['color']})\n")
            
            f.write("\nCOUNT STATISTICS\n")
            f.write("=" * 40 + "\n")
            f.write(f"  Total buildings: {total}\n")
            f.write(f"  Damaged buildings: {damaged} ({damaged/total*100:.1f}%)\n\n")
            for level, info in stats['damage_levels'].items():
                label = legend[level]['label']
                f.write(f"  {label}: {info['count']} ({info['percentage']:.1f}%)\n")
            
            f.write("\nAREA STATISTICS (平方米)\n")
            f.write("=" * 40 + "\n")
            for level in [2, 3, 4]:
                info = area_stats[level]
                label = legend[level]['label']
                if info['count'] > 0:
                    f.write(f"\n  {label}:\n")
                    f.write(f"    数量: {info['count']}\n")
                    f.write(f"    总面积: {info['total_area']:.2f}\n")
                    f.write(f"    最小面积: {info['min_area']:.2f}\n")
                    f.write(f"    最大面积: {info['max_area']:.2f}\n")
                    f.write(f"    平均面积: {info['avg_area']:.2f}\n")
                else:
                    f.write(f"\n  {label}: 0 个建筑\n")
            f.write(f"\n  损毁建筑总面积: {total_area:.2f}\n")
        print(f"  Legend saved: {args.outlegend}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
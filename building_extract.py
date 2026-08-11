import argparse
import os
#rasterio 适合简单遥感影像读写、数据分析
from osgeo import gdal, ogr, osr#栅格转矢量 Polygonize 功能 rasterio 没有原生封装，矢量 SHP 创建、图层、字段、要素遍历全靠 OGR
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import sys
sys.path.append(os.path.dirname(__file__))
from model import SwinUNet

# ====================== 1. 固定全局配置======================
# 输出SHP固定路径（GIS出图脚本直接读取）
OUT_SHP = os.path.join(os.path.dirname(__file__), "output", "building.shp")
# 临时掩码栅格路径
TEMP_MASK_TIF = os.path.join(os.path.dirname(__file__), "output", "temp_build_mask.tif")
# 最小建筑面积（过滤杂草、阴影碎斑块，单位：平方米，墨西哥城区建议设为15㎡）
MIN_BUILD_AREA = 100
# 坐标系固定：WGS84 EPSG:4326（和你的tif完全匹配，杜绝偏移）
TARGET_EPSG = 4326

# ====================== 2. 遥感TIF读取、滑窗推理（适配大图，避免内存溢出）======================
def read_geo_tif(tif_path):
    """读取地理TIF，返回数组、地理仿射、投影、宽高"""
    ds = gdal.Open(tif_path)#读取tif的地理坐标系，仿射变换等
    if ds is None:
        raise FileNotFoundError(f"无法打开影像：{tif_path}")
    width = ds.RasterXSize
    height = ds.RasterYSize
    trans = ds.GetGeoTransform()#获取地理仿射变换六参数（左上角x坐标, 像素x方向分辨率, 旋转系数, 左上角y坐标, 旋转系数, 像素y方向分辨率）
                                #后续保存建筑掩码 tif 时，把这套参数写进去，掩码才能和原图地理位置一模一样
    proj = ds.GetProjection()#获取投影坐标系字符串
    band_num = ds.RasterCount#计影像一共有多少个波段
    # 只读前3个RGB波段（卫星影像通用）
    img = []
    for i in range(min(3, band_num)):#多波段也只读前三个
        band = ds.GetRasterBand(i+1).ReadAsArray(0,0,width,height)#从（0，0）读取整张i+1波段所有参数转numpy二维数组
        img.append(band)
    img = np.stack(img, axis=-1)#波段合并（最后一维合并，hwc日常格式），调整数组维度，适配图片格式
    return img, trans, proj, width, height, ds

def sliding_predict_swinunet(model, img, patch_size=256, stride=192, device="cpu"):
    """SwinUNet 滑窗推理（输入单张图，输出建筑掩码）"""
    
    print(f"img 形状: {img.shape}")
    
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
        print("️ 灰度图已转换为 RGB")
    
    h, w = img.shape[:2]
    print(f"图像尺寸: h={h}, w={w}")
    
    # ✅ 如果图像太小，调整 patch_size
    if h < patch_size or w < patch_size:
        print(f"️ 图像尺寸 ({h}x{w}) 小于 patch_size ({patch_size})")
        patch_size = min(h, w)
        stride = max(1, patch_size // 2)
        print(f"  新的 patch_size: {patch_size}, stride: {stride}")
    
    pred_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    img = img.astype(np.float32) / 255.0
    print("图像归一化完成，开始滑窗推理...")
    
    # ✅ 计算总窗口数
    y_range = range(0, h - patch_size + 1, stride)
    x_range = range(0, w - patch_size + 1, stride)
    y_steps = list(y_range)
    x_steps = list(x_range)
    total_steps = len(y_steps) * len(x_steps)
    print(f"总窗口数: {total_steps}")
    
    if total_steps == 0:
        print(" 总窗口数为 0，检查图像尺寸和 patch_size")
        return np.zeros((h, w), dtype=np.uint8)
    
    step_count = 0
    for y in y_steps:
        for x in x_steps:
            step_count += 1
            if step_count % 10 == 0:
                print(f"  进度: {step_count}/{total_steps}")
            
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            crop = img[y:y_end, x:x_end]
            
            # 填充
            pad_h = patch_size - (y_end - y)
            pad_w = patch_size - (x_end - x)
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            
            if crop.shape[-1] != 3:
                crop = crop[:, :, :3]
            
            crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(crop_tensor)
                pred = torch.sigmoid(out).squeeze().cpu().numpy()
            
            pred = pred[:y_end-y, :x_end-x]
            pred_map[y:y_end, x:x_end] += pred
            count_map[y:y_end, x:x_end] += 1
    
    print(f"推理完成，共处理 {step_count} 个窗口")
    
    pred_map = pred_map / (count_map + 1e-8)
    
    # ... 后面的统计和阈值代码 ...
     # ✅ ===== 打印预测值分布 =====
    pred_flat = pred_map.flatten()
    print("\n" + "=" * 50)
    print("   预测值分布统计:")
    print(f"  总像素数: {len(pred_flat):,}")
    print(f"  最小值: {pred_flat.min():.6f}")
    print(f"  最大值: {pred_flat.max():.6f}")
    print(f"  平均值: {pred_flat.mean():.6f}")
    print(f"  中位数: {np.median(pred_flat):.6f}")
    print(f"  标准差: {pred_flat.std():.6f}")
    print(f"\n  各阈值下建筑像素数:")
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (pred_flat > thresh).sum()
        print(f"    > {thresh}: {count:,}像素（{count/len(pred_flat)*100:.2f}%）")
    print("=" * 50 + "\n")
    if pred_flat.mean() > 0.5:
        threshold = 0.5
    else:
        threshold = 0.3
    pred_bin = (pred_map>threshold).astype(np.uint8)
    print(f"使用阈值{threshold}")
    
    #pred_bin = (pred_map>0.5).astype(np.uint8)
    return pred_bin

def save_mask_tif(mask_arr, ref_trans, ref_proj, out_tif):
    """把推理后的建筑掩码保存为带地理坐标的TIF（二值化）"""
    driver = gdal.GetDriverByName("GTiff")
    h, w = mask_arr.shape
    out_ds = driver.Create(out_tif, w, h, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(ref_trans)
    out_ds.SetProjection(ref_proj)
    band = out_ds.GetRasterBand(1)
    
    # ✅ 确保二值化（0 和 1）
    mask_arr = (mask_arr > 0.5).astype(np.uint8)
    band.WriteArray(mask_arr)
    band.SetNoDataValue(255)  # ✅ 白色作为 NoData
    out_ds.FlushCache()
    del out_ds
    print(f" 掩码已保存（二值化，NoData=255）")

# ====================== 3. 核心：栅格掩码 → SHP矢量 + 过滤小斑块（剔除误识别杂草）======================
import cv2
def raster_to_build_shp(mask_tif, out_shp, min_pixels=80, epsg=4326):
    # 1. 打开掩码栅格
    src_ds = gdal.Open(mask_tif, gdal.GA_Update)
    if src_ds is None:
        print(f" 无法打开掩码栅格：{mask_tif}")
        return
    
    src_band = src_ds.GetRasterBand(1)
    im_geotrans = src_ds.GetGeoTransform()

    # ✅ 把 0 值设为 NoData
    src_band.SetNoDataValue(0)
    
    # ✅ 用 OpenCV 统计每个连通域的像素数
    data = src_band.ReadAsArray()
    data = (data > 0.5).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        data, connectivity=8
    )
    
    # 建立像素数映射：label -> pixel_count
    pixel_count_map = {}
    for i in range(1, num_labels):
        pixel_count_map[i] = stats[i, cv2.CC_STAT_AREA]
    
    print(f"找到 {num_labels - 1} 个连通区域")

    # 2. 定义投影
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg)
    

    # 3. 创建 SHP
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_shp):
        drv.DeleteDataSource(out_shp)
    out_ds = drv.CreateDataSource(out_shp)
    out_layer = out_ds.CreateLayer("buildings", spatial_ref, ogr.wkbPolygon)

    # 4. 创建属性字段
    out_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    out_layer.CreateField(ogr.FieldDefn("area_m2", ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn("px_cnt", ogr.OFTInteger))

    # 5. 栅格转面
    gdal.Polygonize(src_band, src_band, out_layer, 0, [], callback=None)

    # 6. 遍历多边形，用 OpenCV 统计的像素数决定是否保留
    feat_id = 1
    out_layer.ResetReading()
    to_delete = []

    for feat in out_layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            to_delete.append(feat.GetFID())
            continue

        # ✅ 获取几何边界（注意：GetEnvelope 返回元组）
        envelope = geom.GetEnvelope()
        min_x, max_x, min_y, max_y = envelope  # ✅ 直接解包元组

        # 在几何范围内取 5x5 = 25 个采样点
        sample_labels = []
        for i in range(5):
            for j in range(5):
                sample_x = min_x + (max_x - min_x) * (i + 0.5) / 5
                sample_y = min_y + (max_y - min_y) * (j + 0.5) / 5
                
                # 检查点是否在几何内部
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(sample_x, sample_y)
                if point.Within(geom):
                    # 转换为像素坐标
                    px = int((sample_x - im_geotrans[0]) / im_geotrans[1])
                    py = int((sample_y - im_geotrans[3]) / im_geotrans[5])
                    
                    if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
                        sample_labels.append(labels[py, px])
        
        # 取出现次数最多的标签
        if sample_labels:
            from collections import Counter
            label_counter = Counter(sample_labels)
            most_common_label = label_counter.most_common(1)[0][0]
            pixel_count = pixel_count_map.get(most_common_label, 0)
        else:
            # 如果采样点都没命中，用中心点作为备选
            centroid = geom.Centroid()
            px = int((centroid.GetX() - im_geotrans[0]) / im_geotrans[1])
            py = int((centroid.GetY() - im_geotrans[3]) / im_geotrans[5])
            if 0 <= px < data.shape[1] and 0 <= py < data.shape[0]:
                pixel_count = pixel_count_map.get(labels[py, px], 0)
            else:
                pixel_count = 0

        # 过滤碎片
        if pixel_count < min_pixels:
            to_delete.append(feat.GetFID())
            continue
        """
        # 估算面积
        pixel_width = abs(im_geotrans[1])
        pixel_height = abs(im_geotrans[5])
        pixel_area = pixel_width * pixel_height
        geom_area = pixel_count * pixel_area
        """
        
        # ✅ 只用像素数估算面积（不调用 geom.GetArea()）
        # 从影像的地理变换中获取分辨率（度）
        pixel_width_deg = abs(im_geotrans[1])
        pixel_height_deg = abs(im_geotrans[5])
        
        # 1度 ≈ 111320 米（在赤道附近）
        meter_per_degree = 111320
        pixel_width_m = pixel_width_deg * meter_per_degree
        pixel_height_m = pixel_height_deg * meter_per_degree
        area_m2 = pixel_count * pixel_width_m * pixel_height_m
        
        feat.SetField("id", feat_id)
        feat.SetField("area_m2", float(area_m2))
        feat.SetField("px_cnt", int(pixel_count))
        out_layer.SetFeature(feat)
        feat_id += 1
    # 删除不合格要素
    for fid in reversed(to_delete):
        out_layer.DeleteFeature(fid)

    out_ds.Destroy()
    src_ds = None
    print(f"建筑矢量SHP已生成：{out_shp}，有效建筑总数：{feat_id-1}")

# ====================== 4. 程序入口（命令行传参，对接你之前的WorkThread线程）======================
def main():
    parser = argparse.ArgumentParser(description="灾前遥感影像建筑自动提取工具 (SiamUNet)")
    parser.add_argument("--before", required=True, help="灾前遥感TIF路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device} 加载建筑提取模型")

    # ✅ 加载 SwinUNet
    model = SwinUNet(in_channels=3, num_classes=1, img_size=256)
    weight_path = os.path.join(os.path.dirname(__file__), "model", "swinunet_best.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"加载 SwinUNet 权重: {weight_path}")

    # 读取影像
    img, trans, proj, w, h, ds = read_geo_tif(args.before)
    print(f"影像尺寸：{w} × {h}，开始推理")
    
    # 滑窗推理（使用 SwinUNet 的推理函数）
    mask_bin = sliding_predict_swinunet(model, img, patch_size=256, stride=192, device=device)
    # 保存掩码
    save_mask_tif(mask_bin, trans, proj, TEMP_MASK_TIF)

    # 转 SHP
    raster_to_build_shp(TEMP_MASK_TIF, OUT_SHP, MIN_BUILD_AREA, TARGET_EPSG)

    # 清理临时文件
    #if os.path.exists(TEMP_MASK_TIF):
       # os.remove(TEMP_MASK_TIF)

if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "model"), exist_ok=True)
    main()
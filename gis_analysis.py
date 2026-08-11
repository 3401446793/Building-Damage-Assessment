import arcpy
import argparse
import os
import pandas as pd
#读取 AI 输出的损毁栅格，完成栅格转矢量、建筑损毁赋值、乡镇分区统计、导出 Excel 灾情报表
#运行依赖 ArcGIS Pro 自带 Python 环境
#分区统计 + 栅格转矢量
def main():
    parser = argparse.ArgumentParser()#创建参数解析器
    #接收Qt传来的四个外部入参
    parser.add_argument("--tif")#AI 推理输出的损毁分级栅格
    parser.add_argument("--building")#建筑面矢量 shp 路径
    parser.add_argument("--district")#行政区边界 shp 路径
    parser.add_argument("--outgdb")#输出 FileGDB 地理数据库完整路径
    args = parser.parse_args()#解析参数，可通过.参数名读取路径
    arcpy.env.overwriteOutput = True#允许覆盖已存在的矢量 / 栅格文件
    os.makedirs(args.outgdb, exist_ok=True)#创建 GDB 所在文件夹，exist_ok=True文件夹存在也不会报错
    #提取 gdb 所在文件夹，提取result.gdb文件名
    arcpy.CreateFileGDB_management(os.path.dirname(args.outgdb), os.path.basename(args.outgdb))
    #设置当前工作空间为新建的 gdb，后续所有生成矢量（damage_poly、building_damage）
    #都会自动存入这个数据库，不用重复写完整路径
    arcpy.env.workspace = args.outgdb

    # 1. 栅格转面
    #输入推理得的损毁栅格，输出面要素名"damage_poly"存入上面的gdb，不简化轮廓，保留建筑精细边界
    #"VALUE"：栅格的像素值存入矢量属性表字段，字段名就是gridcode，存储像素片区损毁程度
    arcpy.RasterToPolygon_conversion(args.tif, "damage_poly", "NO_SIMPLIFY", "VALUE")
    # 2. 建筑与损毁栅格空间连接
    #根据空间位置，给每一栋建筑匹配覆盖最多的损毁等级，输入原始建筑轮廓面和栅格转换的面，
    #输出带损毁等级的建筑矢量，后续可以单独查询单栋建筑损毁情况，也是分区统计的基础数据。
    arcpy.SpatialJoin_analysis(args.building, "damage_poly", "building_damage")
    # 3. 行政区分区统计
    #输入行政区矢量（乡镇 / 区县边界），行政区唯一名字段用于分组统计，损毁面矢量，输出统计表，存到 GDB 中
    #输出表内置字段
    #NAME：行政区名称
    #COUNT：该行政区内损毁图斑数量
    #AREA：损毁总面积
    #VALUE：损毁等级
    arcpy.ZonalStatisticsAsTable(args.district, "NAME", "damage_poly", "district_stat")
    # 4. 导出Excel统计表
    excel_path = os.path.join(os.path.dirname(args.outgdb), "灾情分区统计表.xlsx")
    data = []
    fields = ["NAME", "COUNT", "AREA", "VALUE"]
    #指定字段生成行政区灾情统计表，建筑的统计也类似
    with arcpy.da.SearchCursor("district_stat", fields) as cur:
        for row in cur:
            data.append(row)
    df = pd.DataFrame(data, columns=["行政区名称", "建筑数量", "损毁面积", "损毁等级"])
    df.to_excel(excel_path, index=False)#不产生多于索引例
    print("GIS统计完成，报表已导出")

if __name__ == "__main__":
    main()
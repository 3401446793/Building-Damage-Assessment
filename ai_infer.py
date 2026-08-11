import argparse
import os
import torch
import rasterio
import numpy as np
from rasterio.windows import Window
from model import SiamUNet,Config  # 网络文件


def predict_with_tta(model, pre_patch, post_patch):#包含不同角度的图像预测
    """8种变换的TTA"""
    transforms = [
        lambda x: x,                                    # 原图
        lambda x: torch.flip(x, dims=[-1]),             # 水平翻转
        lambda x: torch.flip(x, dims=[-2]),             # 垂直翻转
        lambda x: torch.rot90(x, k=1, dims=[-2, -1]),   # 旋转90°
        lambda x: torch.rot90(x, k=2, dims=[-2, -1]),   # 旋转180°
        lambda x: torch.rot90(x, k=3, dims=[-2, -1]),   # 旋转270°
        lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1]),  # 翻转+旋转
        lambda x: torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-1]),  # 翻转+旋转
    ]
    
    preds = []
    #对每种变换进行预测
    for t in transforms:
        #应用变换
        pre_t = t(pre_patch)
        post_t = t(post_patch)
        #模型预测
        with torch.no_grad():
            pred = model(pre_t, post_t)
            pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
            # 逆变换回来（变回原图方向）
            pred = t(torch.from_numpy(pred).float()).numpy()
        preds.append(pred)
    
    # 投票融合（所有预测取平均，再取整），消除锯齿，精确度提升
    return np.round(np.mean(preds, axis=0)).astype(np.int64)


def infer(before_path, after_path, out_path, use_tta=True):
    config = Config()
    model_path = "./model/best_model_epoch12.pth"
    model = SiamUNet(in_channels=config.IN_CHANNELS, 
                        num_classes=config.NUM_CLASSES)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件不存在！，请检查model文件夹内权重文件名")
    model.load_state_dict(torch.load(model_path,map_location="cpu"))
    model.eval()#切换到推理模式

    # 读取图像,只读元信息不读完整大图
    with rasterio.open(before_path) as src:
        h,w = src.height,src.width

    pred_mask = np.zeros((h, w), dtype=np.int64)#初始为0，累加所有预测结果(分类，要取整)
    count = np.zeros((h, w), dtype=np.float32)#初始为0，记录每个像素被预测的次数（重叠区域像素可能被多次预测）
    patch_size = config.PATCH_SIZE
    stride = config.STRIDE
    
    #滑窗推理
    # 读取双时相影像
    with rasterio.open(before_path) as src_pre,rasterio.open(after_path) as src_post:
        profile = src_pre.profile
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                window = Window(x,y,patch_size,patch_size)
                pre_patch = src_pre.read(window=window)
                post_patch = src_post.read(window=window)

                #只读前三个波段
                pre_patch = pre_patch[:3,:,:]#0,1,2通道，全部高，全部宽
                post_patch = post_patch[:3,:,:]

                #归一化
                #pre_patch = pre_patch.astype(np.float32)/255.0
                #post_patch = post_patch.astype(np.float32)/255.0

                # 转tensor
                pre_tensor = torch.from_numpy(pre_patch).float().unsqueeze(0)#最前面（0处）添加batch维度初值1
                post_tensor = torch.from_numpy(post_patch).float().unsqueeze(0)

                #推理
                if use_tta:
                    pred = predict_with_tta(model,pre_tensor,post_tensor)
                else:
                    with torch.no_grad():#无需反向传播，禁用梯度省显存
                        pred = model(pre_tensor, post_tensor)
                        pred = torch.argmax(pred, dim=1).cpu().numpy()[0]#输入为四维的包含每个像素5各通道不同的分的图，
                                                                        #输出为每个像素取最高分三维得到分类+形状的图
                                                                        #【0】去掉输出三维中0号元素即batch，只含h，w
                # 加权累加
                pred_mask[y:y+patch_size, x:x+patch_size] += pred
                count[y:y+patch_size, x:x+patch_size] += 1

    # 取平均（避免除零）
    mask = np.zeros_like(pred_mask)
    valid = count > 0
    mask[valid] = (pred_mask[valid] / count[valid]).astype(np.int64)
    print(" pred_bin 调试:")
    print(f"  pred_bin 唯一值: {np.unique(mask)}")
    print(f"  pred_bin 中 1 的个数: {(mask == 1).sum()}")
    print(f"  pred_bin 中 2 的个数: {(mask == 2).sum()}")
    print(f"  pred_bin 中 3 的个数: {(mask == 3).sum()}")
    print(f"  pred_bin 中 4 的个数: {(mask == 4).sum()}")
    # 输出损毁分级栅格
    profile.update(count=1, dtype=rasterio.uint8)#输出栅格改为单波段
    profile.update(nodata=255)
    with rasterio.open(out_path, "w", **profile) as dst:#写入带地理坐标系的GeoTiff
        dst.write(mask.astype(np.uint8), 1)
    print(f"损毁栅格已输出至: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)#required为真表示该参数必须传入
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--use_tta",
                        action="store_true", help="开启测试时增强推理")#action：命令行里有该参数则值自动为真，否则自动假
                                                                    #python 文件名 -h命令可输出参数和help说明
    args = parser.parse_args()#将参数打包成args对象
    os.makedirs(os.path.dirname(args.out), exist_ok=True)#自动创建输出文件夹
    infer(args.before, args.after, args.out,use_tta=args.use_tta)
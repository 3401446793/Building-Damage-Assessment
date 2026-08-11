
#========================Siam-UNet模型===========================
import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.models.swin_transformer import SwinTransformer

class Config:
    
    # 数据参数（关键：8GB显存建议512）
    PATCH_SIZE = 512
    STRIDE = 384  # 重叠128（20%）
    
    # 多光谱通道数（根据数据修改，RGB是3）
    IN_CHANNELS = 3  # 如果是4通道就改4
    
    # 类别数（背景+4种损毁等级）
    NUM_CLASSES = 5

class SiamUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()#SiamUNet继承自nn。Module，super调用父类的构造函数
        # 共享编码器（ResNet34做骨干）
        self.encoder = resnet34(weights=None)#采用随机权重
        
        # 修改第一层卷积适应多通道（如果是3通道不用改），7x7适合特点稀疏而场景大的遥感图，3x3卷积核适合特征丰富小的图
        if in_channels != 3:
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, #图片二维卷积
                                           kernel_size=7, stride=2, 
                                           padding=3, bias=False)#kernel_size=7大感受野适合遥感，stride下采样2倍减少计算，paddin保持尺寸不变（7-1）/2
                                                                 #卷积偏置冗余bias为false减少参数
        # 获取各层特征，参数由浅入深
        self.enc_layers = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu,
                          self.encoder.maxpool, self.encoder.layer1, 
                          self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]
        
        # 解码器（上采样+跳跃连接）
        self.decoder = UNetDecoder(num_classes)
        
    def forward(self, pre, post, return_aux=False):
        pre_features = self.encode(pre)
        post_features = self.encode(post)
        
        # 拼接 pre 和 post
        concat_features = [
            torch.cat([p, q], dim=1) 
            for p, q in zip(pre_features, post_features)
        ]
        
        return self.decoder(concat_features, return_aux=return_aux)
    
    def encode(self, x):
        features = []
        #提取边缘，颜色                 #64通道ResNet固定
        x = self.encoder.conv1(x)      # 卷积核7x7提取局部特征
        x = self.encoder.bn1(x)         #批归一化，使数据分布稳定
        x = self.encoder.relu(x)        #激活非线性，使模型可学习到更多复杂特征
        features.append(x)              # f0: 64

        #提取纹理，局部形状
        x = self.encoder.maxpool(x)     #2x2最大池化尺寸减半（下采样尺寸减小感受变大）
        x = self.encoder.layer1(x)      # 64通道#残差块3个提取局部形状
        features.append(x)              # f1: 64

        #提取建筑轮廓
        x = self.encoder.layer2(x)      # 128通道#4个残差快
        features.append(x)              # f2: 128

        #提取建筑结构
        x = self.encoder.layer3(x)      # 256通道#6个残差块
        features.append(x)              # f3: 256

        #提取损毁类型
        x = self.encoder.layer4(x)      # 512通道#3个
        features.append(x)              # f4: 512
        
        return features

class UNetDecoder(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super().__init__()
        
        # 解码器：上采样 + 跳跃连接
        # 输入 features 已经是 pre+post 拼接后的
        # features[0]: 128, features[1]: 128, features[2]: 256, features[3]: 512, features[4]: 1024
        
        self.upconv4 = nn.Sequential(#容器块，内部按序执行
            nn.Conv2d(1024 + 512, 512, 3, padding=1),#编码器最底层pre和post拼接后1024个通道，第三层合起来512
                                                    #拼接后1536，下一层接收的输出通道512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.8)
        )
        
        self.upconv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.6)
        )
        
        self.upconv1 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.4)
        )
        #双线性插值放大，恢复原始分辨率无棋盘格
        self.final_upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        #1x1卷积核恢复成5通道类别数
        self.final_conv = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, features, return_aux=False):
        # features: [f0, f1, f2, f3, f4] 从浅到深
        
        x = features[4]  # 最深层，通道数 1024
        
        # 第4层 → 第3层
        #双线性插值，根据周围四个像素值计算新像素值
        x = nn.functional.interpolate(x, size=features[3].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, features[3]], dim=1)  # 1024 + 512 = 1536#通道维度合并相加
        x = self.upconv4(x)  # → 512
        
        # 第3层 → 第2层
        x = nn.functional.interpolate(x, size=features[2].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, features[2]], dim=1)  # 512 + 256 = 768
        x = self.upconv3(x)  # → 256
        
        # 第2层 → 第1层
        x = nn.functional.interpolate(x, size=features[1].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, features[1]], dim=1)  # 256 + 128 = 384
        x = self.upconv2(x)  # → 128
        
        # 第1层 → 第0层
        x = nn.functional.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, features[0]], dim=1)  # 128 + 128 = 256
        x = self.upconv1(x)  # → 64
        x = self.final_upsample(x)#上面定义的函数对象回调
        
        return self.final_conv(x)
    

class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, img_size=256):
        super().__init__()
        self.encoder = SwinTransformer(
            img_size=img_size,
            patch_size=4,
            in_chans=in_channels,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(48, num_classes, 1)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        
        if len(features.shape) == 4:
            features = features.permute(0, 3, 1, 2)
        elif len(features.shape) == 3:
            B, L, C = features.shape
            H = W = int(L ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        #print(f"输入到解码器: {features.shape}")
        
        # 逐层打印
        for i, layer in enumerate(self.decoder):
            features = layer(features)
            #print(f"  layer {i}: {features.shape}")
        
        return features
#========================极简U-Net结构（未使用）===========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #两层3*3卷积等效为一层5*5卷积，视野更大但参数更少速度更快
        #单层 5×5 参数量：\(C×C×5×5=25C^2\)
        #两层 3×3 总参数量：\(C×C×3×3 + C×C×3×3=18C^2\)
        #多一层非线性激活，特征提取能力更强
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #转置卷积将尺寸放大一倍回复分辨率
        #输出通道是输入通道整除除以2，以2为步长2x2大小卷积核卷积
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):#只有1维，是与不是建筑
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inp = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits
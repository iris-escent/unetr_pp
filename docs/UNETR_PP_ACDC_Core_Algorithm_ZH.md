# ACDC 版 UNETR++ 核心算法深度解析

## 1. 这份文档讲什么

这份文档只讲你当前代码库里的 **ACDC 版本网络前向主链路**，目标是帮助你从零开始看懂：

- 输入一个 3D patch 之后，网络每一步到底做了什么
- 编码器、解码器、skip connection、深监督在代码里是怎么落地的
- 每个主要张量的 shape 是怎么变化的
- 当前 ACDC 版本为什么和“原始 UNETR++ 论文结构”不完全一样

这份文档 **不展开**：

- 数据预处理
- patch 采样
- 损失函数实现
- 滑动窗口推理

---

## 2. 版本说明：你现在看的不是论文原版 UNETR++

先把一个很重要的前提讲清楚：

- 原始 UNETR++ 常见讲法里，`TransformerBlock` 核心通常会强调 `EPA (Efficient Paired Attention)`
- 但你当前 ACDC 分支里的 `TransformerBlock` 已经不是那个原版 EPA 实现
- 当前 ACDC 版本的核心块更接近：

```text
MobileMamba token 路径
+ 3D 卷积局部路径
+ Mamba-ScConv cross-attention 融合
+ 残差投影
```

也就是说：

- **原始 UNETR++** 更像“分层编码器 + Transformer/EPA + U 形解码器”
- **当前 ACDC 版本** 是“分层编码器 + Mamba/Conv/Fusion 混合块 + U 形解码器 + 边缘增强”

所以你读代码时要始终记住：

- 有些术语是 UNETR++ 的通用概念
- 但当前具体实现细节，必须以 ACDC 目录下的代码为准

---

## 3. 最小阅读地图

如果你想顺着代码读，先盯住这几个文件：

- 主网络入口：
  `unetr_pp/network_architecture/acdc/unetr_pp_acdc.py`
- 编码器和解码器组件：
  `unetr_pp/network_architecture/acdc/model_components.py`
- 当前 ACDC 版核心块：
  `unetr_pp/network_architecture/acdc/transformerblock.py`
- 融合模块：
  `unetr_pp/network_architecture/mamba_scconv_fusion.py`
- MobileMamba 子模块：
  `unetr_pp/network_architecture/mobile_mamba_block.py`
- ScConv 子模块：
  `unetr_pp/network_architecture/ScConv.py`
- 训练器里真实输入配置：
  `unetr_pp/training/network_training/unetr_pp_trainer_acdc.py`

文档里默认使用训练器中的真实输入设置：

```text
input_res = (1, 16, 160, 160)
```

这里表示：

- `C = 1`：输入通道数，ACDC 通常是单通道 MRI
- `D = 16`：深度方向切出的 patch 厚度
- `H = 160`
- `W = 160`

如果加上 batch 维度，真正进网络的是：

```text
[B, 1, 16, 160, 160]
```

本文统一采用张量顺序：

```text
[B, C, D, H, W]
```

注意：代码里有时把后三维变量名写成 `H, W, D`，但本质上仍然是在处理一个 3D 体特征图。阅读时重点看数值变化，不必被变量名困住。

---

## 4. 一张图先看完整前向链路

### 4.1 主干总览

```text
输入 x_in: [B, 1, 16, 160, 160]
│
├─ 浅层卷积分支 encoder1
│  └─ convBlock: [B, 16, 16, 160, 160]
│
├─ 边缘分支 edge_extractor
│  └─ full_edge_map: [B, 8, 16, 160, 160]
│
└─ 分层编码器 unetr_pp_encoder
   │
   ├─ stem downsample: (1,4,4) stride
   │  └─ [B, 32, 16, 40, 40]
   ├─ stage 0: 3 个 TransformerBlock
   │  └─ enc1 = [B, 32, 16, 40, 40]
   │
   ├─ downsample 1: (2,2,2) stride
   │  └─ [B, 64, 8, 20, 20]
   ├─ stage 1: 3 个 TransformerBlock
   │  └─ enc2 = [B, 64, 8, 20, 20]
   │
   ├─ downsample 2: (2,2,2) stride
   │  └─ [B, 128, 4, 10, 10]
   ├─ stage 2: 3 个 TransformerBlock
   │  └─ enc3 = [B, 128, 4, 10, 10]
   │
   ├─ downsample 3: (2,2,2) stride
   │  └─ [B, 256, 2, 5, 5]
   ├─ stage 3: 3 个 TransformerBlock
   │  └─ [B, 256, 2, 5, 5]
   │
   └─ flatten 成 token
      └─ enc4 = [B, 50, 256]

解码阶段
enc4 --proj_feat--> dec4: [B, 256, 2, 5, 5]
dec4 --decoder5 + skip(enc3)--> dec3: [B, 128, 4, 10, 10]
dec3 --decoder4 + skip(enc2) + edge_4--> dec2: [B, 64, 8, 20, 20]
dec2 --decoder3 + skip(enc1) + edge_3--> dec1: [B, 32, 16, 40, 40]
dec1 --decoder2 + skip(convBlock)--> out: [B, 16, 16, 160, 160]

输出头
out  --out1--> logits1: [B, K, 16, 160, 160]
dec1 --out2--> logits2: [B, K, 16, 40, 40]
dec2 --out3--> logits3: [B, K, 8, 20, 20]
```

这里的 `K` 是类别数，也就是 `out_channels`。

### 4.2 主网络 forward 伪代码

当前 ACDC 主网络前向流程可以压缩成下面这样：

```text
1. hidden_states = encoder(x_in)
2. convBlock = encoder1(x_in)
3. full_edge_map = edge_extractor(x_in)
4. enc1, enc2, enc3, enc4 = hidden_states
5. dec4 = proj_feat(enc4)
6. dec3 = decoder5(dec4, enc3)
7. dec2 = decoder4(dec3, enc2, edge_4)
8. dec1 = decoder3(dec2, enc1, edge_3)
9. out  = decoder2(dec1, convBlock)
10. 输出 deep supervision logits
```

这已经能看出它是一个很典型的 U 形结构：

- 左边编码器：不断下采样，压缩空间，提取抽象特征
- 右边解码器：不断上采样，恢复分辨率
- 中间用 skip connection 把浅层细节送回解码器

---

## 5. 先补几个零基础概念

### 5.1 什么是 feature map

`feature map` 可以先理解成：

- 原始图像经过网络变换后得到的“特征图”
- 不再是直接的像素灰度，而是网络学出来的语义表示

例如：

```text
[B, 1, 16, 160, 160]
```

是原始输入；

```text
[B, 32, 16, 40, 40]
```

就是 32 个通道的特征图。每个通道都在表达某种模式，比如边缘、局部组织结构、纹理组合、器官形状线索等。

### 5.2 什么是 encoder / decoder

- `encoder`：把图像逐渐压缩成更抽象、更全局的特征
- `decoder`：把这些抽象特征重新放大，恢复到接近原图分辨率，用来做像素级或体素级预测

### 5.3 什么是 downsampling / upsampling

- `downsampling`：让空间尺寸变小，比如 `160 -> 40`
- `upsampling`：让空间尺寸变大，比如 `20 -> 40`

空间变小的好处：

- 感受野更大
- 计算量更可控
- 高层语义更容易形成

### 5.4 什么是 skip connection

skip connection 可以先用一句话理解：

**把前面某一层的特征，直接送到后面更深的地方。**

在这个网络里，skip connection 主要体现在 decoder 中：

```python
out = self.transp_conv(inp)
out = out + skip
```

意思是：

- 先把深层特征上采样
- 再和编码器同尺度特征相加

这样做的原因是：

- 深层特征语义强，但细节粗
- 浅层特征细节多，但语义弱
- 把它们融合后，解码器更容易恢复准确边界

### 5.5 什么是 residual connection

residual connection 就是：

```text
y = x + f(x)
```

它的意思不是“凭空多加一步”，而是：

- 保留原始输入 `x`
- 再加上一条变换支路 `f(x)`

这样网络训练更稳定，因为模型不需要每次都“推翻重来”，而是在原表示上做增量修正。

### 5.6 什么是 token

在 Transformer 或 Mamba 风格模块里，常常不直接处理 `[B, C, D, H, W]`，而是把空间位置展平成一串序列：

```text
[B, C, D, H, W] -> [B, N, C]
```

其中：

- `N = D * H * W`
- 每个空间位置对应一个 token
- 每个 token 的维度是通道数 `C`

所以 token 不是“额外造出来的神秘对象”，它其实就是：

**把 3D 特征图中的每个空间位置，当成一个序列元素。**

---

## 6. 编码器：分层下采样到底在做什么

编码器类是 `UnetrPPEncoder`，定义在：

```text
unetr_pp/network_architecture/acdc/model_components.py
```

### 6.1 编码器整体思想

编码器没有一上来就把整张 3D patch 直接展平成超长序列，而是：

1. 先用卷积做第一次 patch embedding
2. 在某个分辨率上做若干个块提特征
3. 再下采样到更小分辨率
4. 再做若干个块
5. 重复这个过程

这就叫 **分层编码**。

它的直觉是：

- 分辨率高时，保留更多细节
- 分辨率低时，更适合建模全局结构
- 多个尺度同时保留，后面解码器更容易恢复结果

### 6.2 当前 ACDC 版的编码器配置

训练器里真实配置是：

```text
feature_size = 16
dims         = [32, 64, 128, 256]
depths       = [3, 3, 3, 3]
num_heads    = 4
```

这里：

- `dims` 表示 4 个 stage 的通道数
- `depths` 表示每个 stage 里堆多少个 `TransformerBlock`

所以这个编码器其实是：

```text
stage 0: 通道 32，块数 3
stage 1: 通道 64，块数 3
stage 2: 通道 128，块数 3
stage 3: 通道 256，块数 3
```

---

## 7. 第一层：为什么说它是“卷积式 patch embedding”

### 7.1 代码操作

编码器第一层 `stem_layer` 是：

```python
get_conv_layer(..., kernel_size=(1, 4, 4), stride=(1, 4, 4))
```

后面再接一个 group norm。

### 7.2 输入输出 shape

输入：

```text
[B, 1, 16, 160, 160]
```

输出：

```text
[B, 32, 16, 40, 40]
```

### 7.3 为什么尺寸会这样变

因为这层卷积：

- 在深度方向 `D` 上：`kernel=1, stride=1`
- 在平面方向 `H/W` 上：`kernel=4, stride=4`

所以：

- `D` 不变：`16 -> 16`
- `H` 变成 `160 / 4 = 40`
- `W` 变成 `160 / 4 = 40`
- 通道从 `1 -> 32`

### 7.4 为什么这叫 patch embedding

如果你把这层卷积想成“切块”，它等价于：

- 每次看一个 `1 x 4 x 4` 的局部块
- 步长也是 `1 x 4 x 4`
- 因此相邻两次卷积窗口几乎不重叠

所以你可以把它理解成：

**把原图划成许多小 patch，每个 patch 映射成一个 32 维特征。**

只是这里没有手工 `for` 循环切 patch，而是用一次 stride 卷积把：

- 切 patch
- patch 内线性映射
- 输出通道扩展

这三件事一起做了。

所以它叫 **卷积式 patch embedding**。

### 7.5 为什么先压 H/W，不先压 D

这是 3D 医学图像中很常见的设计直觉：

- 体数据的深度方向往往更薄、层数更少
- 平面方向 `H/W` 往往更大
- 一开始就压缩 `H/W`，可以大幅降低计算
- 先不压 `D`，能尽量保住切片间的信息

对于当前输入：

```text
16 x 160 x 160
```

如果第一层就把 `D` 也一并大幅压掉，前期可能会损失过多 3D 结构信息。

---

## 8. 后三层下采样：为什么像“合并相邻特征块”

后面三层 downsample 都是：

```python
kernel_size=(2, 2, 2), stride=(2, 2, 2)
```

### 8.1 第 2 个尺度

输入：

```text
[B, 32, 16, 40, 40]
```

输出：

```text
[B, 64, 8, 20, 20]
```

变化是：

- `16 -> 8`
- `40 -> 20`
- `40 -> 20`
- `32 -> 64`

### 8.2 第 3 个尺度

输入：

```text
[B, 64, 8, 20, 20]
```

输出：

```text
[B, 128, 4, 10, 10]
```

### 8.3 第 4 个尺度

输入：

```text
[B, 128, 4, 10, 10]
```

输出：

```text
[B, 256, 2, 5, 5]
```

### 8.4 为什么说这是“相邻 2x2x2 特征块合并”

因为：

- 卷积核是 `2x2x2`
- 步长也是 `2x2x2`

这意味着每次卷积会看一个局部的 3D 小立方块，再把它映射成新的、更抽象的特征。

你可以把它想象成：

- 原来 8 个相邻小体素位置，各自有自己的特征
- 下采样后，这 8 个位置合成 1 个新位置
- 新位置表示的是更大区域的摘要信息

### 8.5 为什么“空间变小、通道变多”

这是深度网络里很常见的设计：

- 空间尺寸变小后，特征图更粗
- 为了补偿信息容量，通常会增加通道数

粗略理解就是：

- “横向位置数量”减少了
- “每个位置能表达的语义维度”增加了

所以：

- 浅层更像“高分辨率、低语义”
- 深层更像“低分辨率、高语义”

---

## 9. 每个 stage 为什么要先提特征，再继续下采样

编码器不是：

```text
downsample -> downsample -> downsample -> downsample
```

而是：

```text
downsample -> stage blocks -> downsample -> stage blocks -> ...
```

每个 stage 都会堆 `depths[i] = 3` 个 `TransformerBlock`。

这样做的原因是：

1. 下采样只负责改变尺度，不负责充分建模
2. 同一分辨率上堆叠多个 block，能在当前尺度充分提特征
3. 不同尺度擅长不同类型信息

例如：

- 大尺寸特征图更适合保留局部边界和细节
- 小尺寸特征图更适合看整体形状和长程关系

所以“先处理、再压缩”，比“一路只压尺寸”更合理。

---

## 10. 编码器 forward：hidden_states 到底存了什么

编码器前向逻辑可以概括成：

```text
x = downsample0(x)
x = stage0(x)
hidden_states.append(x)

x = downsample1(x)
x = stage1(x)
hidden_states.append(x)

x = downsample2(x)
x = stage2(x)
hidden_states.append(x)

x = downsample3(x)
x = stage3(x)
x = flatten_to_tokens(x)
hidden_states.append(x)
```

所以：

- `hidden_states[0] = enc1 = [B, 32, 16, 40, 40]`
- `hidden_states[1] = enc2 = [B, 64, 8, 20, 20]`
- `hidden_states[2] = enc3 = [B, 128, 4, 10, 10]`
- `hidden_states[3] = enc4 = [B, 50, 256]`

其中最后一级为什么是 `[B, 50, 256]`？

因为：

```text
2 * 5 * 5 = 50
```

也就是把最后的 3D 特征图：

```text
[B, 256, 2, 5, 5]
```

展平为 50 个 token，每个 token 维度是 256。

---

## 11. `TransformerBlock`：这是当前 ACDC 版的真正核心

当前 ACDC 版的 `TransformerBlock` 定义在：

```text
unetr_pp/network_architecture/acdc/transformerblock.py
```

它的输入输出都是：

```text
[B, C, D, H, W]
```

但中间会临时转成序列：

```text
[B, N, C]
```

### 11.1 一张内部数据流图

```text
x: [B, C, D, H, W]
│
├─ reshape + permute
│  └─ x_seq: [B, N, C]
│
├─ 加位置编码 pos_embed
│
├─ LayerNorm
│
├─ MobileMambaBlockSeq
│
├─ gamma 缩放后与输入残差相加
│  └─ x_mamba_seq: [B, N, C]
│
├─ reshape 回 3D
│  └─ x_mamba_3d: [B, C, D, H, W]
│
├─ conv51
├─ conv52
│  └─ conv_feat: [B, C, D, H, W]
│
├─ fusion(x_mamba_3d, conv_feat)
│  └─ fused: [B, C, D, H, W]
│
├─ conv8(fused)
│
└─ 输出 out = fused + conv8(fused)
   └─ [B, C, D, H, W]
```

### 11.2 第一步：3D 特征图变成 token 序列

代码：

```text
x.reshape(B, C, D*H*W).permute(0, 2, 1)
```

把：

```text
[B, C, D, H, W]
```

变成：

```text
[B, N, C]
```

其中：

```text
N = D * H * W
```

例如在第一个 stage：

```text
[B, 32, 16, 40, 40]
```

会变成：

```text
[B, 25600, 32]
```

这一步的含义是：

- 把空间位置拉直成一个序列
- 让后面的序列模块可以在“位置序列”上工作

### 11.3 第二步：加位置编码 `pos_embed`

如果不用位置编码，单独看 token 序列时，模块很难知道：

- 第 1 个 token 原来在左上角
- 第 2 个 token 原来在器官中心
- 第 3 个 token 原来在边界附近

所以会加一个和序列同形状的可学习参数：

```text
[1, N, C]
```

然后做：

```text
x_seq = x_seq + pos_embed
```

直觉上就是：

**在每个 token 自身特征之外，再额外告诉模型“我在空间中的哪个位置”。**

### 11.4 第三步：LayerNorm 到底在做什么

代码是：

```text
self.norm(x_seq)
```

对于单个 token 的通道向量 `x_i ∈ R^C`，LayerNorm 会在最后一维上做归一化：

```text
μ_i = (1/C) * Σ_j x_{i,j}
σ_i^2 = (1/C) * Σ_j (x_{i,j} - μ_i)^2
LN(x_{i,j}) = γ_j * (x_{i,j} - μ_i) / sqrt(σ_i^2 + ε) + β_j
```

你可以这样理解：

- 一个 token 有很多通道值
- 不同通道尺度可能差很多
- LayerNorm 先把这个 token 的各通道数值拉到更稳定的范围

这样后面的序列模块更容易训练。

### 11.5 第四步：`gamma` 残差缩放

当前实现里有：

```text
x_mamba_seq = x_seq + gamma * mobile_mamba_block(norm(x_seq))
```

这就是一个残差结构：

```text
y = x + α f(x)
```

其中：

- `x` 是原输入
- `f(x)` 是 Mamba 路径算出来的修正量
- `alpha = gamma`

为什么要乘 `gamma`？

- 让网络一开始更保守
- 避免一上来就让新分支改动过猛
- 训练时再慢慢学会“该改多少”

### 11.6 第五步：`MobileMambaBlockSeq`

`MobileMambaBlockSeq` 定义在：

```text
unetr_pp/network_architecture/mobile_mamba_block.py
```

它内部不是严格意义上的完整 Mamba 原始实现，而是一个简化后的序列混合块。

它的核心思路是：

1. 先对 token 序列做 LayerNorm
2. 按通道把特征分成三段：
   - global 部分
   - local 部分
   - identity 部分
3. global 和 local 两部分分别过自己的前馈子模块
4. 再拼回去
5. 再过一次线性投影
6. 外层再接一个 FFN 残差

如果输入是：

```text
[B, N, C]
```

输出仍然是：

```text
[B, N, C]
```

直觉上它是在做：

- 一部分通道专门学更“全局”的混合
- 一部分通道专门学更“局部”的混合
- 另一部分通道保留原样

虽然实现已经简化，但它仍然承担了“token 序列级别建模”的角色。

### 11.7 第六步：reshape 回 3D 特征图

经过 token 序列模块后，还要回到卷积世界：

```text
[B, N, C] -> [B, C, D, H, W]
```

这样后面才能接 3D 卷积和融合模块。

### 11.8 第七步：两层 3D 卷积残差块

当前实现有：

```text
conv51 = UnetResBlock(...)
conv52 = UnetResBlock(...)
```

这两层负责的是 **局部空间细节增强**。

为什么要有这条卷积分支？

因为纯 token/序列建模擅长：

- 长程关系
- 全局依赖

但医学图像分割还非常依赖：

- 边界
- 邻域连续性
- 小结构的局部纹理

而卷积在这些局部模式上通常更强。

所以这块设计的直觉是：

- token 路径看“远”
- conv 路径看“近”

### 11.9 第八步：融合模块 `fusion`

在当前 ACDC 文件中，卷积分支不是简单加回去，而是送入：

```text
MambaScConvCrossAttnFusion
```

也就是说它希望更复杂地融合：

- token 路径的信息
- 局部卷积路径的信息

这个模块后面会单独拆开。

### 11.10 第九步：最后一个残差投影 `conv8`

最后输出是：

```text
out = fused + conv8(fused)
```

这仍然是残差结构。

作用是：

- 再做一次轻量修正
- 给输出增加一个可学习的线性/局部投影补偿

---

## 12. 为什么说 token 路径负责长程关系，卷积路径负责局部细节

这句话很常见，但如果不落到数据形状上会很空。

### 12.1 token 路径

把 3D 特征图展平成：

```text
[B, N, C]
```

以后，每个 token 都代表一个空间位置。

序列模块的优势是：

- 可以在“位置序列”层面做更灵活的交互
- 更容易建模远距离位置之间的关系

例如：

- 左心室和右心室某些结构在空间上不相邻
- 但它们可能在更高层语义上存在对应关系

序列/注意力/Mamba 风格模块更适合做这类“跨远距离”的信息传播。

### 12.2 卷积路径

卷积天生是在局部窗口上运算。

3D 卷积擅长：

- 看邻域
- 看边缘连续性
- 看局部小纹理
- 看短程空间一致性

所以分割任务里经常会保留卷积路径，避免模型只会“看大局”，却丢掉边界细节。

---

## 13. `MambaScConvCrossAttnFusion` 逐路径拆解

这个模块定义在：

```text
unetr_pp/network_architecture/mamba_scconv_fusion.py
```

它接收两个输入：

```text
x_mamba: [B, C, D, H, W]
x_conv : [B, C, D, H, W]
```

输出：

```text
out: [B, C, D, H, W]
```

它可以分成三条路径：

1. spatial path
2. channel path
3. cross-attention path

---

## 14. spatial path：关注空间位置上的融合

### 14.1 输入与第一步拼接

先把两个分支按通道拼接：

```text
cat_spatial = concat(x_mamba, x_conv, dim=channel)
```

shape：

```text
[B, 2C, D, H, W]
```

### 14.2 1x1x1 卷积投影到 `d`

接着用：

```text
self.s_q_conv
```

把通道数从 `2C` 投影到 `d`：

```text
q_emb: [B, d, D, H, W]
```

这里的直觉是：

- 先把两路特征混在一起
- 再为每个空间位置提取一个较低维的“空间嵌入”

### 14.3 再投影回 `C`

然后经过：

```text
self.s_out
```

得到：

```text
spatial_out: [B, C, D, H, W]
```

这可以理解成：

- 从“融合后的空间嵌入”里
- 生成一张新的空间增强特征图

### 14.4 `alpha_s` 门控

然后做：

```text
spatial_fused = alpha_s * x_mamba + (1 - alpha_s) * spatial_out
```

这里：

- `alpha_s = sigmoid(raw_alpha_s)`
- shape 是 `[1, C, 1, 1, 1]`

它表示：

- 对每个通道，模型自己决定
- 是更相信原来的 `x_mamba`
- 还是更相信新生成的 `spatial_out`

所以 `alpha_s` 是一个 **可学习门控系数**。

---

## 15. channel path：关注通道信息的重整

### 15.1 为什么要专门讲“通道”

通道维本质上表示不同类型的语义特征。

例如：

- 某些通道更偏边缘
- 某些通道更偏区域内部
- 某些通道更偏器官轮廓

所以除了空间位置之间的关系，模型还会关心：

**哪些通道应该被强化，哪些通道应该被抑制。**

### 15.2 先把 3D 特征拆成按切片的 2D 特征

代码中有：

```text
x_conv.permute(...).reshape(B * D, C, H, W)
```

这一步把 3D 特征按深度切成很多 2D slice，送入 `ScConv`：

```text
[B, C, D, H, W] -> [B*D, C, H, W]
```

### 15.3 `ScConv` 在做什么

`ScConv` 定义在 `ScConv.py`，内部主要是：

```text
SRU -> CRU
```

可以先用直觉理解它：

- `SRU`：通过门控把信息分成更重要和次重要部分，再重组
- `CRU`：对通道进行分组、压缩、卷积、融合

它的目的不是单纯增加层数，而是：

- 让通道结构更有选择性
- 提升局部纹理和通道重排能力

### 15.4 再拼回 3D

`ScConv` 输出后，再恢复回：

```text
[B, C, D, H, W]
```

### 15.5 通道融合

然后做：

```text
cat_channel = concat(x_mamba, x_sc)
channel_out = channel_reduce(cat_channel)
```

shape 变化：

```text
[B, 2C, D, H, W] -> [B, C, D, H, W]
```

### 15.6 `alpha_c` 门控

然后做：

```text
channel_fused = alpha_c * x_mamba + (1 - alpha_c) * channel_out
```

含义和 `alpha_s` 类似：

- 每个通道自己学会“更相信旧特征还是新通道融合特征”

---

## 16. cross-attention path：让空间查询通道摘要

这一部分最容易让初学者卡住，所以慢一点讲。

### 16.1 先解释 query / key / value

在注意力里可以先这样理解：

- `query`：我现在想问什么
- `key`：我这里有什么可供匹配的信息标签
- `value`：一旦匹配上，我真正要取回的内容

一句很粗略的话：

```text
query 去找最相关的 key，然后把对应的 value 取回来
```

### 16.2 当前代码里谁是 query

这里用的是空间路径生成的 `q_emb`：

```text
q = q_emb.view(B, d, N).permute(0, 2, 1)
```

shape：

```text
[B, N, d]
```

意思是：

- 每个空间位置都有一个 query 向量
- 空间位置总数是 `N = D*H*W`

也就是：

**每个空间位置都在问：我应该从通道摘要里拿什么信息回来？**

### 16.3 当前代码里谁是 key/value

先对 `channel_fused` 做全局平均池化：

```text
ch_gap = channel_fused.mean(dim=(2, 3, 4))
```

shape：

```text
[B, C]
```

这一步就是 **global average pooling**。

含义是：

- 把每个通道在整个空间上的响应压成一个标量
- 得到“每个通道总体上有多强”的摘要

然后：

```text
ch_tokens = Linear(ch_gap.unsqueeze(-1))
```

变成：

```text
[B, C, d]
```

所以这里：

- 序列长度不是空间位置数 `N`
- 而是通道数 `C`

也就是：

**每个通道被当成一个 token。**

这些通道 token 同时作为：

- `key`
- `value`

### 16.4 cross-attention 真正在算什么

PyTorch 调用是：

```text
attn_out, _ = cross_mha(q, ch_tokens, ch_tokens)
```

即：

```text
Q = q         ∈ R^{B×N×d}
K = ch_tokens ∈ R^{B×C×d}
V = ch_tokens ∈ R^{B×C×d}
```

注意力核心可写成：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
```

直觉上意思是：

- 每个空间位置的 query
- 去看哪些通道 token 和自己更匹配
- 再把这些通道 token 的信息加权取回

这就是为什么我会把它描述成：

**让空间位置去查询通道摘要。**

### 16.5 reshape 回 3D 并映射回通道维

attention 输出先是：

```text
[B, N, d]
```

再变回：

```text
[B, d, D, H, W]
```

然后通过 `attn_to_C`：

```text
[B, d, D, H, W] -> [B, C, D, H, W]
```

### 16.6 最终 `gamma` 门控

最后：

```text
out = gamma * spatial_fused + (1 - gamma) * attn_proj
```

这一步在做最终融合：

- 一部分相信 spatial_fused
- 一部分相信 cross-attention 结果

所以 `gamma` 的作用也是可学习门控。

---

## 17. 解码器：为什么能把小特征图还原回分割结果

解码器组件在：

```text
unetr_pp/network_architecture/acdc/model_components.py
```

核心类是：

```text
UnetrUpBlock
```

### 17.1 解码器每一级在干什么

每一级 `UnetrUpBlock` 大致做：

```text
1. 转置卷积上采样
2. 与 skip 特征融合
3. 如果有边缘特征，再融合边缘特征
4. 再过当前尺度的 block
```

代码主干就是：

```text
out = transp_conv(inp)
out = out + skip
if edge_feat is not None:
    out = concat(out, edge_feat)
    out = edge_fusion_conv(out)
out = decoder_block(out)
```

### 17.2 什么是转置卷积上采样

转置卷积可以先粗略理解成：

**一种可学习的上采样方式。**

它和普通插值不一样：

- 普通插值通常是固定规则
- 转置卷积有可学习参数

所以网络能自己学：

- 如何把低分辨率特征恢复到高分辨率

### 17.3 为什么 `out = out + skip` 就是 skip connection

因为 `skip` 来自编码器同尺度特征。

例如：

- decoder 某层上采样后变成 `8 x 20 x 20`
- encoder 某层恰好也有 `8 x 20 x 20`

这时把两者相加，就是把浅层细节“跳接”回来了。

这就是 skip connection 在代码里的具体实现。

### 17.4 为什么这里用“相加”而不是“拼接”

很多 U-Net 用的是：

```text
concat(decoder_feat, encoder_feat)
```

但这里用的是：

```text
add(decoder_feat, encoder_feat)
```

用相加的特点是：

- 显存更省
- 通道数不暴涨
- 更像残差融合

代价是：

- 要求两边 shape 和通道严格对齐

当前这套网络正是按这种对齐方式设计的。

---

## 18. 解码器主链路逐级展开

### 18.1 `dec4 = proj_feat(enc4, ...)`

编码器最后给出的 `enc4` 是 token 形式：

```text
[B, 50, 256]
```

解码器想继续用 3D 方式处理，所以先恢复成：

```text
[B, 256, 2, 5, 5]
```

这一步不是上采样，只是：

- token 序列重新排回 3D 网格

### 18.2 `dec3 = decoder5(dec4, enc3)`

输入：

- `dec4 = [B, 256, 2, 5, 5]`
- `enc3 = [B, 128, 4, 10, 10]`

步骤：

1. `dec4` 先转置卷积上采样
2. 得到 `[B, 128, 4, 10, 10]`
3. 和 `enc3` 相加
4. 再过 decoder block

输出：

```text
dec3 = [B, 128, 4, 10, 10]
```

### 18.3 `dec2 = decoder4(dec3, enc2, edge_4)`

输入：

- `dec3 = [B, 128, 4, 10, 10]`
- `enc2 = [B, 64, 8, 20, 20]`
- `edge_4 = [B, 8, 8, 20, 20]`

步骤：

1. `dec3` 上采样为 `[B, 64, 8, 20, 20]`
2. 与 `enc2` 相加
3. 再把边缘特征拼接上去
4. 用 `1x1x1` 卷积把通道压回 64
5. 过当前 decoder block

输出：

```text
dec2 = [B, 64, 8, 20, 20]
```

### 18.4 `dec1 = decoder3(dec2, enc1, edge_3)`

输入：

- `dec2 = [B, 64, 8, 20, 20]`
- `enc1 = [B, 32, 16, 40, 40]`
- `edge_3 = [B, 8, 16, 40, 40]`

输出：

```text
dec1 = [B, 32, 16, 40, 40]
```

### 18.5 `out = decoder2(dec1, convBlock)`

输入：

- `dec1 = [B, 32, 16, 40, 40]`
- `convBlock = [B, 16, 16, 160, 160]`

这里的上采样核是：

```text
(1, 4, 4)
```

所以会把：

```text
16 x 40 x 40
```

恢复到：

```text
16 x 160 x 160
```

然后与浅层卷积分支 `convBlock` 相加，再过最后一个卷积解码块，得到：

```text
out = [B, 16, 16, 160, 160]
```

---

## 19. 边缘分支：为什么要额外引入 LaplacianEdgeExtractor3D

当前 ACDC 版本里，在主编码器之外还有一条边缘分支：

```text
full_edge_map = edge_extractor(x_in)
```

这不是原始 UNETR++ 的必备部分，而是当前代码里的额外增强。

### 19.1 拉普拉斯核做什么

拉普拉斯算子本质上是在检测：

- 强烈的灰度变化
- 高频边缘

如果某个位置周围变化很剧烈，它的响应就会更大。

所以这条分支是在告诉网络：

**图像里哪些地方更像边界。**

### 19.2 为什么要取绝对值

拉普拉斯响应可能有正有负：

- 正值和负值都可能代表边缘变化

如果只看符号，容易丢掉“强变化”的强度信息，所以这里做：

```text
raw_edge = abs(raw_edge)
```

强调的是边缘强度，而不是正负方向。

### 19.3 为什么还要 `InstanceNorm3d + 1x1x1 Conv`

原始边缘响应往往：

- 数值波动大
- 很尖锐
- 直接送主干可能不稳定

所以作者又做了：

1. `InstanceNorm3d`
   - 把不同样本的数值尺度拉平
2. `1x1x1 Conv`
   - 把单通道硬边缘映射成多通道软特征
3. 再 `InstanceNorm3d + LeakyReLU`

这意味着边缘分支不是只输出“死板的硬边缘”，而是输出：

**适合和主干特征融合的软边缘表示。**

### 19.4 边缘特征如何对齐 decoder 尺度

完整边缘图是：

```text
[B, 8, 16, 160, 160]
```

但不同 decoder 阶段需要不同分辨率，所以代码用了：

```text
F.adaptive_avg_pool3d(full_edge_map, output_size=enc2.shape[2:])
F.adaptive_avg_pool3d(full_edge_map, output_size=enc1.shape[2:])
```

得到：

- `edge_4 = [B, 8, 8, 20, 20]`
- `edge_3 = [B, 8, 16, 40, 40]`

这样就能和对应 decoder 层对齐。

### 19.5 为什么边缘特征不是直接相加，而是先拼接再卷积

代码里是：

```text
out = torch.cat([out, edge_feat], dim=1)
out = edge_fusion_conv(out)
```

原因是：

- `out` 是主干特征
- `edge_feat` 是边缘提示

两者语义不完全一样，直接相加太生硬。

先拼接，再用 `1x1x1` 卷积融合，能让网络自己学习：

- 哪些边缘信息该保留
- 哪些该弱化

---

## 20. 输出头与 deep supervision

主网络最后会输出：

```text
if do_ds:
    [out1(out), out2(dec1), out3(dec2)]
else:
    out1(out)
```

### 20.1 三个输出分别对应什么

- `out1(out)`：
  最终最高分辨率输出
  shape 是 `[B, K, 16, 160, 160]`
- `out2(dec1)`：
  中间辅助输出
  shape 是 `[B, K, 16, 40, 40]`
- `out3(dec2)`：
  更深层辅助输出
  shape 是 `[B, K, 8, 20, 20]`

### 20.2 什么是 deep supervision

deep supervision 就是：

**不只监督最终输出，还同时监督中间层输出。**

这样做常见的好处是：

- 让梯度更容易传到中间层
- 让解码器不同尺度都学会分割
- 训练更稳定

### 20.3 为什么推理时常常只用最终输出

因为最终真正需要的是最高分辨率的预测结果。

所以训练时多头一起学，推理或验证时通常只保留最终头。这也是训练器里会临时关闭 `do_ds` 的原因。

---

## 21. 从输入到输出，把整条链再口语化复述一遍

如果你暂时不想纠结公式，可以把整个网络先记成下面这段话：

1. 输入一个 3D MRI patch
2. 一条浅层卷积分支保留高分辨率细节
3. 主编码器先把图像切成更粗的 patch 特征，再一级一级下采样
4. 每一级下采样后，都用混合块提特征
5. 最深层特征被展平成 token，再在解码时重新排回 3D
6. 解码器逐级上采样
7. 每一级都把对应 encoder 的特征通过 skip connection 加回来
8. 中间两级还融合边缘分支给出的边界提示
9. 最后输出分割 logits
10. 训练时额外输出两个辅助头做 deep supervision

---

## 22. 术语词典：每个名词都落到这份代码里

### 22.1 patch

通俗定义：
从整张 3D 医学图像里裁出来的一小块体数据。

在医学分割里的作用：
整卷太大，显存不够，训练时通常用 patch 而不是整卷。

在当前代码里怎么实现：
训练器真实输入 shape 是 `[B, 1, 16, 160, 160]`，这就是一个 3D patch。

### 22.2 patch embedding

通俗定义：
把原始 patch 变成网络更容易处理的特征表示。

在医学分割里的作用：
把原始像素/体素信号变成高维语义特征。

在当前代码里怎么实现：
编码器第一层 `(1,4,4)` stride 卷积把 `[B,1,16,160,160]` 变成 `[B,32,16,40,40]`。

### 22.3 token

通俗定义：
把一个空间位置当成一个序列元素。

在医学分割里的作用：
让序列模块、注意力模块、Mamba 模块处理空间位置关系。

在当前代码里怎么实现：
`[B,C,D,H,W] -> [B,N,C]`，其中 `N=D*H*W`。

### 22.4 stage

通俗定义：
一个固定分辨率下的一整组处理模块。

在医学分割里的作用：
不同 stage 表示不同尺度的特征提取。

在当前代码里怎么实现：
每个 stage 是“一个 downsample 后接 3 个 TransformerBlock”。

### 22.5 encoder

通俗定义：
逐步压缩空间、提取抽象语义的部分。

在医学分割里的作用：
形成多尺度语义特征。

在当前代码里怎么实现：
`UnetrPPEncoder`。

### 22.6 decoder

通俗定义：
逐步放大特征、恢复空间分辨率的部分。

在医学分割里的作用：
把高层语义还原成逐体素分割结果。

在当前代码里怎么实现：
`decoder5 -> decoder4 -> decoder3 -> decoder2`。

### 22.7 downsampling

通俗定义：
把特征图尺寸缩小。

在医学分割里的作用：
扩大感受野，减少计算，提取更高层语义。

在当前代码里怎么实现：
`stride=(1,4,4)` 和三层 `stride=(2,2,2)` 卷积。

### 22.8 upsampling

通俗定义：
把特征图尺寸放大。

在医学分割里的作用：
恢复细粒度空间预测。

在当前代码里怎么实现：
`UnetrUpBlock` 中的转置卷积。

### 22.9 skip connection

通俗定义：
把前面层的特征直接送到后面层。

在医学分割里的作用：
补回浅层边界和细节。

在当前代码里怎么实现：
`out = out + skip`。

### 22.10 residual connection

通俗定义：
输出等于输入加上一个修正量。

在医学分割里的作用：
让网络更稳定、更容易训练。

在当前代码里怎么实现：
例如 `x + gamma * mobile_mamba_block(...)`，以及 `fused + conv8(fused)`。

### 22.11 normalization

通俗定义：
把特征数值拉到更稳定的范围。

在医学分割里的作用：
稳定训练，改善梯度传播。

在当前代码里怎么实现：
包括 `LayerNorm`、`GroupNorm`、`InstanceNorm3d`、`BatchNorm3d`。

### 22.12 positional embedding

通俗定义：
给 token 加上“位置信息提示”。

在医学分割里的作用：
让序列模块知道不同 token 原来位于哪里。

在当前代码里怎么实现：
`self.pos_embed`，shape 为 `[1, N, C]`，直接加到 token 序列上。

### 22.13 bottleneck

通俗定义：
网络最深、最小分辨率的那一层。

在医学分割里的作用：
承载最抽象、最全局的语义特征。

在当前代码里怎么实现：
`[B,256,2,5,5]` 以及其 token 形式 `[B,50,256]`。

### 22.14 feature map

通俗定义：
网络内部的特征图。

在医学分割里的作用：
表示不同层次的图像语义。

在当前代码里怎么实现：
比如 `enc1`、`enc2`、`dec1`、`dec2` 都是 feature map。

### 22.15 channel

通俗定义：
特征维度，每个通道表示一种特征响应。

在医学分割里的作用：
容纳不同类型的语义模式。

在当前代码里怎么实现：
例如 `32 / 64 / 128 / 256` 这些就是不同 stage 的通道数。

### 22.16 spatial

通俗定义：
空间位置维度，也就是体数据里的位置坐标。

在医学分割里的作用：
决定“哪里是哪里”。

在当前代码里怎么实现：
`D, H, W` 维度就是 spatial 维。

### 22.17 deep supervision

通俗定义：
训练时对中间层也进行监督。

在医学分割里的作用：
帮助多尺度学习，稳定优化。

在当前代码里怎么实现：
`[out1(out), out2(dec1), out3(dec2)]`。

### 22.18 logits

通俗定义：
分类头输出的原始分数，还没过 softmax。

在医学分割里的作用：
后续用于计算损失或转成概率图。

在当前代码里怎么实现：
`UnetOutBlock` 的输出就是 logits。

---

## 23. 第一次读这份代码，最容易卡住的点

### 23.1 变量名里的维度顺序

代码里有时会写 `h w d`，有时你脑中会想 `d h w`。不要被名字带偏，先盯 shape 数值变化。

### 23.2 `enc4` 为什么突然变成 `[B, 50, 256]`

因为最后一级编码器被 flatten 成 token 了，不再是 3D feature map。

### 23.3 为什么 `x_output` 看起来没怎么用

主网络里更主要使用的是 `hidden_states`，尤其是最后的 `enc4` 和前几级 skip 特征。

### 23.4 为什么当前 ACDC 块和资料里的 EPA 讲法对不上

因为你当前分支已经换成了 ACDC 自己的混合块实现，不是论文原版的 EPA block。

### 23.5 为什么有的地方是相加，有的地方是拼接

- skip connection：通常是相加
- 边缘特征融合：先拼接再 `1x1x1` 卷积

两者功能不同，不要混为一谈。

---

## 24. 推荐阅读顺序

如果你准备真正把代码啃下来，建议按这个顺序：

1. 先看 `unetr_pp_acdc.py` 的 `forward`
2. 再看 `model_components.py` 里的 `UnetrPPEncoder`
3. 再看 `UnetrUpBlock`
4. 再看 `acdc/transformerblock.py`
5. 最后看 `mamba_scconv_fusion.py`
6. 如果还想追细节，再看 `mobile_mamba_block.py` 和 `ScConv.py`

第一次读时不要一上来就钻 `fusion` 细节，否则很容易迷路。

正确顺序应该是：

```text
先看大框架 -> 再看 shape 流 -> 再看单个模块 -> 最后看复杂融合
```

---

## 25. 一句话总总结

当前 ACDC 版 UNETR++ 可以把它记成：

**一个分层 3D 编码器-解码器网络，左边用卷积式 patch embedding 和多尺度下采样提特征，中间每个 stage 用 Mamba/卷积/跨注意力融合块增强表示，右边再通过上采样、skip connection 和边缘特征融合把空间分辨率恢复回来，最终输出多尺度分割 logits。**


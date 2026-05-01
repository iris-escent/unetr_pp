# UNETR++ 代码库技术文档

这份文档面向“刚开始系统读医学图像分割工程代码”的同学，目标不是只告诉你某个函数做了什么，而是帮你建立一张完整的心智地图：

1. 这个仓库为什么会看起来这么大。
2. 一个任务（这里以 ACDC 为例）从原始 NIfTI 数据到训练、验证、推理、生成结果到底经历了什么。
3. `UNETR++` 真正的核心创新点在代码里是怎么体现出来的。
4. 训练、验证、推理、patch、滑动窗口这些术语，在医学图像分割工程里各自具体指什么。

---

## 1. 先给你一张全局地图

这个仓库并不是“单纯的 UNETR++ 模型代码”，而是三层东西叠在一起：

1. `nnU-Net/nnFormer` 风格的数据工程框架  
   负责数据目录组织、裁剪、重采样、归一化、plans 文件、patch 采样、滑动窗口推理、结果回写、评估等。
2. `Trainer` 训练器层  
   负责把“数据流、损失函数、优化器、训练循环、验证逻辑、checkpoint 保存”串起来。
3. `UNETR++` 网络结构层  
   负责把一个 3D patch 输入网络，经过编码器、注意力块、解码器后输出分割 logits。

如果你现在最大的困惑是“为什么这么多文件”，本质原因就是：

- 做医学图像分割，不只是“写个模型 forward”。
- 工程上还要解决体数据裁剪、重采样、patch 采样、类别不平衡、测试时整图拼接、原始空间恢复、NIfTI 导出、Dice/HD95 评估、后处理这些问题。

所以最好的阅读方式不是一个文件一个文件从头看到尾，而是先按“主流程”看，再回头看细节。

---

## 2. 读这个仓库时最重要的两个事实

### 2.1 这个仓库明显继承了 `nnFormer/nnU-Net` 的工程骨架

你会在代码里频繁看到这些痕迹：

- `Task001_ACDC`、`Task002_Synapse` 这种任务命名。
- `plans.pkl / plans_3D.pkl` 这种实验规划文件。
- `unetr_pp_raw_data_base`、`unetr_pp_preprocessed`、`RESULTS_FOLDER` 这些环境变量。
- `DataLoader3D`、`crop`、`preprocess`、`validate`、`postprocessing` 这些典型 nnU-Net 工作流。

也就是说，理解这个项目最好的方法，不是把它当成“只有 Transformer 的论文代码”，而是把它当成：

“一个 `nnU-Net` 风格训练框架 + 一个替换进去的 `UNETR++` 主干网络”。

### 2.2 你当前工作区里的 ACDC 代码，不完全等于论文原版 UNETR++

这是非常重要的结论。

在你当前工作区里，`ACDC` 分支已经有明显的本地改动：

- `unetr_pp/network_architecture/acdc/unetr_pp_acdc.py`
  引入了 `LaplacianEdgeExtractor3D` 边缘分支。
- `unetr_pp/network_architecture/acdc/transformerblock.py`
  使用了 `MobileMambaBlockSeq` 和 `MambaScConvCrossAttnFusion`，这已经不是论文原始 EPA block 的最简实现。
- `unetr_pp/training/network_training/unetr_pp_trainer_acdc.py`
  把 `max_num_epochs` 改成了 `5`，注释里写了“本机测试”。

所以你阅读时要区分两件事：

1. “这个仓库当前 ACDC 分支实际在跑什么”。
2. “论文里的 UNETR++ 核心创新原始长什么样”。

如果你的目标是理解论文的原始核心思想，建议优先看：

- `unetr_pp/network_architecture/synapse/unetr_pp_synapse.py`
- `unetr_pp/network_architecture/synapse/model_components.py`
- `unetr_pp/network_architecture/synapse/transformerblock.py`

这三份代码更接近论文里“标准 UNETR++ + EPA”的表达。

如果你的目标是理解“你现在手上的这个项目到底怎么跑”，那就必须看 `acdc/` 目录的实际实现。

---

## 3. 仓库结构总览

下面先按目录解释每一块是干什么的。

### 3.1 根目录

- `README.md`  
  项目介绍、数据组织方式、训练和评估脚本入口。
- `requirements.txt`  
  依赖列表。
- `training_scripts/`  
  各数据集训练启动脚本。
- `evaluation_scripts/`  
  各数据集评估启动脚本。
- `media/`  
  论文图片与结果示意图。
- `output_acdc/`  
  当前工作区里已有的 ACDC 训练输出与验证结果。
- `DATASET_Acdc/`  
  当前工作区里的 ACDC 数据。
- `格式转化.py`  
  独立脚本，看名字像是个人数据转换辅助脚本，不属于主训练框架主干。

### 3.2 `unetr_pp/`

这是主 Python 包，所有核心逻辑都在这里。

- `run/`  
  训练入口与默认配置选择。
- `training/`  
  训练器、dataloader、数据增强、loss、优化器、模型恢复。
- `network_architecture/`  
  模型结构定义。
- `preprocessing/`  
  裁剪、重采样、归一化、预处理。
- `experiment_planning/`  
  分析数据集并生成 plans。
- `inference/`  
  推理与分割结果导出。
- `evaluation/`  
  Dice、HD95 等评估。
- `postprocessing/`  
  连通域后处理。
- `utilities/`  
  常用工具函数。
- `paths.py`  
  环境变量路径管理。
- `configuration.py`  
  全局线程数、各向异性阈值等配置。

---

## 4. 你最该先抓住的主流程

如果把整个项目压缩成一句话，就是：

`原始 NIfTI -> 裁剪/重采样/归一化 -> 生成 stage 数据 -> 随机采样 patch -> 数据增强 -> 网络前向 -> 计算损失并反传 -> patch 级验证指标 -> 训练结束后整卷滑窗验证 -> 评估 Dice/HD95 -> 导出 NIfTI 结果`

下面我们沿着 ACDC 任务，把这条链完整走一遍。

---

## 5. ACDC 任务：从原始数据到最终结果的完整过程

这一节是整份文档最重要的部分。

---

## 5.1 原始数据长什么样

以当前工作区为例，ACDC 数据在：

- `DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC`

里面有：

- `imagesTr/`：训练图像
- `labelsTr/`：训练标签
- `imagesTs/`：测试图像
- `labelsTs/`：测试标签
- `dataset.json`：数据集描述

`dataset.json` 里定义了：

- 这是 MRI 单模态数据。
- 类别共有 4 个标签：`0` 背景，`1` 右心室腔，`2` 心肌，`3` 左心室腔。
- `numTraining=160`
- `numTest=40`

也就是说，最原始的 ACDC 样本本质上是：

- 输入：一个 3D 心脏 MRI 体数据 `patientXXX_frameYY.nii.gz`
- 标签：一个同尺寸的 3D 分割标签 `patientXXX_frameYY_gt.nii.gz`

---

## 5.2 从原始数据到 cropped data：先裁掉无用背景

### 入口代码

- `unetr_pp/experiment_planning/nnFormer_plan_and_preprocess.py`
- `unetr_pp/experiment_planning/utils.py`
- `unetr_pp/preprocessing/cropping.py`

### 关键函数

- `crop(task_name, ...)`
- `ImageCropper.run_cropping(...)`
- `crop_to_nonzero(...)`

### 这一步在做什么

医学图像里，原始体数据通常有大量黑背景。直接整卷训练：

- 浪费显存
- 浪费计算
- patch 采样时很容易抽到全背景

所以第一步先做“非零区域裁剪”。

核心逻辑在 `crop_to_nonzero(data, seg)`：

1. 在所有模态上找非零 mask。
2. 根据 mask 计算包围盒 `bbox`。
3. 对图像和标签一起裁剪到这个 bbox。
4. 被裁掉的纯背景区域，在标签里会被标成 `-1`。

这里 `-1` 很关键，它表示：

“这部分是因为裁剪而丢掉的外部区域，不应该被当成正常背景监督。”

后续数据增强里会用 `RemoveLabelTransform(-1, 0)` 把它转换处理掉。

### 裁剪后的文件形式

裁剪后，每个 case 会保存成：

- `xxx.npz`
- `xxx.pkl`

其中：

- `npz['data']` 通常是 `[图像通道..., 分割标签]` 拼在一起
- `pkl` 记录元信息，比如原始 spacing、原始 shape、裁剪 bbox、ITK 方向信息等

对应目录类似：

- `DATASET_Acdc/unetr_pp_raw/unetr_pp_cropped_data/Task001_ACDC`

---

## 5.3 DatasetAnalyzer：统计数据集特征

### 入口代码

- `unetr_pp/experiment_planning/DatasetAnalyzer.py`

### 它做什么

在真正决定网络 patch size、batch size、目标 spacing 之前，框架会先分析整个数据集。

`DatasetAnalyzer.analyze_dataset()` 主要统计：

- 每个样本裁剪后的大小
- 每个样本原始 spacing
- 所有类别列表
- 各类别体积大小
- 各模态的强度分布统计
- 裁剪后体积缩小了多少

这些统计会写入：

- `dataset_properties.pkl`

后面 experiment planner 就靠它来决定：

- 重采样到什么 spacing
- patch 多大
- 该做几次 pooling
- batch size 设多少

---

## 5.4 Experiment Planner：决定 patch size、batch size、stage

### 入口代码

- `unetr_pp/experiment_planning/experiment_planner_baseline_3DUNet_v21.py`
- `unetr_pp/experiment_planning/experiment_planner_baseline_3DUNet.py`

### 关键函数

- `ExperimentPlanner3D_v21.get_target_spacing()`
- `ExperimentPlanner3D_v21.get_properties_for_stage()`
- `run_preprocessing()`

### 这一步在做什么

这个阶段会生成所谓的 `plans` 文件。你可以把它理解成：

“这个数据集该怎么被网络吃进去的一份配方表”。

配方表里会写：

- `patch_size`
- `batch_size`
- `current_spacing`
- `num_pool_per_axis`
- `pool_op_kernel_sizes`
- `conv_kernel_sizes`
- `normalization_schemes`
- `data_identifier`
- `transpose_forward / transpose_backward`

### ACDC 的特殊性

ACDC 往往在 z 轴上分辨率很低、层数也不多，所以它是强各向异性数据。

这会影响：

- 目标 spacing 的选择
- 是否把某个轴单独处理
- patch 的形状为什么会是类似 `[16, 160, 160]` 这种“薄片厚、平面大”的形式

这也是为什么你在 ACDC 的 trainer 里会看到：

- `patch_size = [16, 160, 160]`

这不是随便拍脑袋定的，而是数据集性质 + 显存预算 + 规划策略共同决定的。

---

## 5.5 Preprocessing：重采样、归一化、生成 stage 数据

### 入口代码

- `unetr_pp/preprocessing/preprocessing.py`
- `unetr_pp/experiment_planning/experiment_planner_baseline_3DUNet.py`

### 关键函数

- `GenericPreprocessor.resample_and_normalize(...)`
- `resample_patient(...)`
- `resample_data_or_seg(...)`
- `GenericPreprocessor._run_internal(...)`

### 具体做了什么

对每个 case，预处理流程大致是：

1. 读取裁剪后的 `npz + pkl`
2. 根据 `transpose_forward` 调整轴顺序
3. 按 `plans['current_spacing']` 重采样图像和标签
4. 做强度归一化
5. 统计前景类别位置 `class_locations`
6. 保存新的 stage 数据

### 为什么要重采样

不同病人的 MRI spacing 可能不同。

如果不统一 spacing：

- 相同尺寸的 patch 实际代表的物理大小不同
- 网络学到的“器官大小感”会混乱

所以框架会把所有病例都 resample 到统一 spacing。

### 为什么标签和图像的插值方式不同

图像是连续值，用高阶插值没问题。  
标签是类别编号，不能插值成 1.37、2.81 这种值，所以标签通常用最近邻或分割专用插值。

### 为什么要记录 `class_locations`

`GenericPreprocessor._run_internal()` 会为每个类别保存一批前景体素坐标：

- 这样 `DataLoader3D` 就可以“有意识地”从前景附近采样 patch
- 否则很多 patch 都会是全背景

这就是后面 `oversample_foreground_percent=0.33` 能成立的基础。

### 这一步输出到哪里

预处理后，会出现类似目录：

- `.../Task001_ACDC/nnFormerData_plans_v2.1_stage0`

注意：从 `paths.py` 和 `default_configuration.py` 的命名来看，这个项目后来想把前缀统一到 `unetr_pp_*`，但你当前工作区里 ACDC 的实际预处理目录名仍然保留了旧的 `nnFormerData_plans_v2.1_stage0`。这属于历史兼容痕迹，不代表你看错了。

里面每个 case 仍然是：

- `xxx.npz`
- `xxx.pkl`

但这次的 `npz` 已经是：

- 裁剪过
- 重采样过
- 归一化过
- 可直接喂训练的 stage 数据

---

## 5.6 训练入口：脚本如何把一切串起来

### 当前 ACDC 训练脚本

- `training_scripts/run_training_acdc.sh`

它做了三件核心事情：

1. 设置环境变量
   - `RESULTS_FOLDER=../output_acdc`
   - `unetr_pp_preprocessed=.../Task01_ACDC`
   - `unetr_pp_raw_data_base=.../unetr_pp_raw`
2. 指定网络类型 `3d_fullres`
3. 指定 trainer 类 `unetr_pp_trainer_acdc`

最终调用：

```bash
python ../unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_acdc 1 0
```

这 4 个位置参数的含义是：

1. `3d_fullres`
   表示训练 3D 全分辨率模型。
2. `unetr_pp_trainer_acdc`
   表示用 ACDC 专属 trainer。
3. `1`
   表示任务 ID 1，也就是 ACDC。
4. `0`
   表示训练第 0 折。

---

## 5.7 `run_training.py`：训练真正开始的地方

### 文件

- `unetr_pp/run/run_training.py`

### 它做了什么

`run_training.py` 是整个训练流程的总调度入口：

1. 解析命令行参数
2. 把 `task id=1` 转成 `Task001_ACDC`
3. 调用 `get_default_configuration(...)`
4. 找到合适的 trainer 类
5. 实例化 trainer
6. `trainer.initialize()`
7. `trainer.run_training()`
8. 训练结束后再调用 `trainer.validate()`

也就是说，`run_training.py` 自己不负责“训练细节”，它只负责：

“根据任务和 trainer 名，把正确的组件拼起来”。

---

## 5.8 `get_default_configuration()`：决定用哪个 plans、哪个输出目录

### 文件

- `unetr_pp/run/default_configuration.py`

### 核心输出

这个函数会返回：

- `plans_file`
- `output_folder_name`
- `dataset_directory`
- `batch_dice`
- `stage`
- `trainer_class`

对 ACDC 而言，含义分别是：

- 从哪个 `plans_3D.pkl` 读配置
- 训练结果存到哪里
- 预处理数据目录在哪
- Dice 是按 batch 还是按 sample 聚合
- 使用第几个 stage
- 具体用哪个 trainer 类

### 你当前工作区的一个特殊点

当前 ACDC 路径结构不是最标准的 nnU-Net 目录，而是“外层 `Task01_ACDC`，内层 `Task001_ACDC`”。

所以这里实际会形成这样的组合：

- `unetr_pp_preprocessed` 指向 `.../Task01_ACDC`
- `convert_id_to_task_name(1)` 找到 `Task001_ACDC`
- `dataset_directory = join(preprocessing_output_dir, task)`

最终得到：

- `.../Task01_ACDC/Task001_ACDC`

这也是为什么你会在路径里同时看到 `Task01` 和 `Task001`。

---

## 5.9 `Trainer_acdc` 与 `unetr_pp_trainer_acdc`：训练逻辑的核心

### 文件关系

- `training/network_training/network_trainer_acdc.py`
  抽象训练循环框架。
- `training/network_training/Trainer_acdc.py`
  ACDC 的通用 trainer，定义数据加载、验证、评估等。
- `training/network_training/unetr_pp_trainer_acdc.py`
  真正使用 UNETR++ 网络的 ACDC trainer。

### 三层职责分工

`NetworkTrainer_acdc`

- 提供通用训练循环
- 管理 epoch、loss、checkpoint、patience、mixed precision

`Trainer_acdc`

- 指定 ACDC 的数据集加载、split、验证导出、在线 Dice 统计
- 指定默认 loss：`DC_and_CE_loss`

`unetr_pp_trainer_acdc`

- 替换网络为 `UNETR_PP`
- 打开 deep supervision
- 配置数据增强
- 改写 patch size、pooling、优化器等

---

## 5.10 数据是怎么从磁盘进到训练循环里的

### 关键文件

- `training/dataloading/dataset_loading.py`
- `training/data_augmentation/default_data_augmentation.py`
- `training/data_augmentation/data_augmentation_moreDA.py`

### 第一步：`load_dataset()`

`load_dataset(folder)` 不会立刻把所有体数据读进内存，而是先建立一个索引字典：

- 每个 case 对应一个 `data_file`
- 一个 `properties_file`

这样可以避免一开始爆内存。

### 第二步：`do_split()`

`Trainer_acdc.do_split()` 使用 `KFold(n_splits=5)` 自动生成：

- `train` case 列表
- `val` case 列表

注意：

- 你现在运行的是 `fold_0`
- 所以只是在 5 折中的第 0 折上训练与验证

### 第三步：`DataLoader3D.generate_train_batch()`

这一步是你最需要理解的。

它不是“整卷读入训练”，而是：

1. 随机选一些 case
2. 从每个 case 中切一个 3D patch
3. 把这些 patch 组成 batch

伪代码可以写成：

```text
for each batch:
    随机抽 batch_size 个病例
    对每个病例:
        如果要前景过采样:
            从 class_locations 里挑一个前景体素
            让 patch 尽量覆盖这个前景体素
        否则:
            随机裁一个 patch
        如果 patch 超出图像边界:
            做 padding
    返回 data patch 和 seg patch
```

### 为什么要 patch 训练

原因非常实际：

- 3D 医学影像太大，整卷输入很容易爆显存
- patch 训练可以增加样本多样性
- 可以通过前景过采样缓解类别不平衡

所以这里的“一个 batch”，不是几个完整病人，而是几个从病人体数据里切出来的小块。

---

## 5.11 数据增强在做什么

### ACDC trainer 用的是

- `get_moreDA_augmentation(...)`

也就是较强的数据增强版本。

### 常见增强包括

- 随机旋转
- 随机缩放
- 高斯噪声
- 高斯模糊
- 亮度变化
- 对比度增强
- 低分辨率模拟
- Gamma 变化
- 翻转镜像

### 一个非常重要的点

数据增强是在 patch 级别做的，不是在整卷级别做的。

也就是说训练真正喂给网络的不是“原始 patch”，而是：

`DataLoader3D 采样 patch -> augmentation pipeline 变换 -> Tensor`

---

## 5.12 训练循环：每一步到底发生什么

### 核心代码

- `NetworkTrainer_acdc.run_training()`
- `NetworkTrainer_acdc.run_iteration()`

### 单次 iteration

`run_iteration()` 的逻辑非常清晰：

1. 从 generator 里取一批 `data` 和 `target`
2. 转成 torch tensor
3. 放到 GPU
4. `output = self.network(data)`
5. `loss = self.loss(output, target)`
6. 如果是训练模式，就 `backward()` + `optimizer.step()`
7. 如果要求在线评估，就统计 Dice 所需的 TP/FP/FN

伪代码：

```text
batch = next(data_generator)
data, target = batch["data"], batch["target"]
output = network(data)
loss = criterion(output, target)
if train:
    loss.backward()
    optimizer.step()
if online_eval:
    统计当前 batch 的 Dice 相关计数
```

### 一个 epoch 在做什么

`run_training()` 里每个 epoch 做：

1. 训练若干个 batch
2. 记录训练 loss
3. 在验证集 patch 上跑若干个 batch
4. 记录验证 loss
5. 汇总在线 Dice
6. 更新学习率
7. 判断是否保存 best checkpoint
8. 判断是否早停

---

## 5.13 Loss：为什么会有 deep supervision

### 关键文件

- `training/loss_functions/dice_loss.py`
- `training/loss_functions/deep_supervision.py`

### 基础 loss

`Trainer_acdc` 默认 loss 是：

- `DC_and_CE_loss`

也就是：

- Dice loss
- Cross-Entropy loss

二者求和。

这样做的动机是：

- Dice 对类别不平衡更敏感
- CE 对逐 voxel 分类更稳定

### deep supervision 是什么

`unetr_pp_trainer_acdc` 把 `self.deep_supervision = True`

于是网络不只输出最终分割图，还会输出中间分辨率的辅助分割头：

- `out1`：最终输出
- `out2`：中间层输出
- `out3`：更浅一层输出

然后 `MultipleOutputLoss2` 会对这些输出分别算 loss，并按权重加和。

伪代码：

```text
logits = [final_pred, mid_pred_1, mid_pred_2]
loss = 0.57 * loss(final_pred, gt0)
     + 0.29 * loss(mid_pred_1, gt1)
     + 0.14 * loss(mid_pred_2, gt2)
```

这样做的好处是：

- 浅层特征也能直接得到监督
- 梯度传播更稳定
- 编码器和解码器多层都能学到有用分割信息

---

## 5.14 训练中的“验证”到底是什么

这是初学者最容易混淆的地方。

这个项目里其实有两种验证。

### 第一种：训练循环里的 patch 级验证

位置：

- `run_training()` 里的 `self.run_iteration(self.val_gen, False, True)`

特点：

- 用的是验证集 patch
- 不回传梯度
- 主要目的是监控 loss 和在线 Dice
- 这个结果会参与 `model_best` 的保存判断

注意：

- 这里不是整卷验证
- 这里不是最终论文表格里的结果

### 第二种：训练结束后的整卷验证

位置：

- `run_training.py` 最后调用 `trainer.validate()`
- `Trainer_acdc.validate()`

特点：

- 对验证集中的每个完整病例做整卷推理
- 用滑动窗口把整卷分块预测
- 把 softmax resample 回原始尺寸
- 导出 NIfTI
- 再和 GT 做 Dice/HD95 等评估

最终会在：

- `validation_raw/summary.json`

得到真正的体级评估结果。

所以你可以把两种验证理解成：

1. 训练中 patch 级验证：用来调训练过程。
2. 训练后整卷验证：用来出正式结果文件。

---

## 5.15 `model_best` 和 `model_final` 到底有什么区别

这是你提到的另一个核心困惑。

### `model_best.model`

保存位置来自：

- `NetworkTrainer_acdc.manage_patience()`

触发条件是：

- 当前 `val_eval_criterion_MA` 超过历史最好值

这里的 `val_eval_criterion_MA` 不是最终 `summary.json` 的整卷 Dice，而是训练时在线验证指标的移动平均。

所以：

- `best` 是“训练过程中 patch 级验证指标最优”的 checkpoint
- 不一定等于最终整卷验证表现最优

### `model_final_checkpoint.model`

保存位置来自：

- `NetworkTrainer_acdc.run_training()`

触发条件是：

- 训练循环结束后，无条件保存最后一轮模型

所以：

- `final` 是最后时刻的权重
- `best` 是训练过程中某个更优时刻的权重

### 为什么有时只有 final，没有 best

在你当前工作区的 ACDC 输出目录里，我只看到了：

- `model_final_checkpoint.model`

没有看到：

- `model_best.model`

这通常有几种可能：

1. 训练过程中没有触发“超过历史最好值”的条件。
2. 训练设置被改得太短，比如当前 ACDC trainer 里 `max_num_epochs=5`，很可能来不及形成稳定提升。
3. 这个输出目录来自测试性训练，不是完整正式训练。

结合你当前代码，我更倾向于第 2、3 种。

---

## 5.16 为什么当前工作区结果看起来不太像论文结果

从代码和已有输出可以看出两个现实情况：

1. `unetr_pp_trainer_acdc.py` 里把 `max_num_epochs` 改成了 `5`
2. ACDC 网络结构已经被改成了带 Mamba、ScConv、边缘分支的实验版本

这意味着：

- 当前工作区更像“个人实验工程”
- 不像“论文原版可直接复现的官方干净分支”

当前已有的 `validation_raw/summary.json` 里，很多前景类别 Dice 为 0，这也说明这个输出目录更像测试训练产物，而不是正式收敛模型。

所以如果你拿这个工作区直接对照论文数字，会非常容易困惑。

---

## 5.17 训练结束后，结果是怎么得到的

训练结束后，`run_training.py` 会调用：

- `trainer.validate()`

对验证集每个病例做整卷预测。

流程是：

1. 读取验证病例的预处理数据 `npz`
2. 去掉最后一层标签通道，只保留图像通道
3. 调用 `predict_preprocessed_data_return_seg_and_softmax()`
4. 网络内部执行滑动窗口推理
5. 得到 softmax 概率图
6. 概率图转回原始轴顺序
7. 调用 `save_segmentation_nifti_from_softmax()` 恢复原始空间并写成 NIfTI
8. 生成 `validation_raw/*.nii.gz`
9. 用 `aggregate_scores()` 评估并写入 `summary.json`
10. 再跑后处理，生成 `validation_raw_postprocessed/`

所以“结果是怎么得出来的”的答案是：

- 不是直接拿训练时 patch 的输出算的
- 而是训练后重新对整个验证病例做一次完整推理，再和 GT 对比得出来的

---

## 6. 推理阶段：滑动窗口是怎么实现的

这也是你特别关心的部分。

### 关键文件

- `unetr_pp/network_architecture/neural_network.py`
- `unetr_pp/inference/predict.py`
- `unetr_pp/inference/segmentation_export.py`

### 为什么需要滑动窗口

整张 3D MRI 体数据通常比训练 patch 大得多。

例如网络训练时只见过：

- `16 x 160 x 160`

但完整病人的体数据可能更大，不能整卷一次性喂进去。

于是就要：

1. 把整卷切成多个重叠 patch
2. 每个 patch 单独预测
3. 再把所有 patch 的结果拼回整卷

这就叫滑动窗口推理。

### 代码入口

`Trainer_acdc.predict_preprocessed_data_return_seg_and_softmax()`

内部会调：

- `self.network.predict_3D(...)`

接着进入：

- `SegmentationNetwork._internal_predict_3D_3Dconv_tiled(...)`

### 核心步骤

#### 1. 先 padding

如果图像尺寸比 patch 小，就先 pad 到至少 patch 大小。

#### 2. 计算滑窗步长

调用：

- `_compute_steps_for_sliding_window(patch_size, image_size, step_size)`

如果 `step_size=0.5`，意思是每次移动 patch 尺寸的一半。

例如：

- patch 宽 160
- step_size 0.5

则相邻窗口大约重叠 80 个 voxel。

#### 3. 逐 patch 预测

三层循环：

```text
for x in steps_x:
    for y in steps_y:
        for z in steps_z:
            取出当前 patch
            做一次网络预测
            累加到整卷结果图上
```

#### 4. 可选 Gaussian 加权

如果 `use_gaussian=True`：

- patch 中心位置的预测权重更高
- patch 边缘位置的预测权重更低

原因是：

- patch 边界处通常预测不如中心稳定

实现上会先生成一个 Gaussian importance map，再对每个 patch 预测结果加权。

#### 5. 可选 Test Time Augmentation

`do_mirroring=True` 时，会做镜像测试增强：

- 原图预测一次
- 沿 x/y/z 翻转后再预测若干次
- 再翻转回来取平均

代码在：

- `_internal_maybe_mirror_and_pred_3D(...)`

### 滑窗推理的伪代码

```text
pad_volume_if_needed()
steps = compute_steps(patch_size, image_size, step_size=0.5)
result = zeros(num_classes, full_volume_shape)
count_map = zeros(num_classes, full_volume_shape)

for each sliding window position:
    patch = volume[x1:x2, y1:y2, z1:z2]
    pred_patch = model(patch)
    if tta:
        pred_patch = average(pred over mirrored versions)
    if gaussian:
        pred_patch *= gaussian_weight
    result[window] += pred_patch
    count_map[window] += gaussian_weight_or_ones

prob = result / count_map
seg = argmax(prob)
```

这就是医学图像分割里“滑动窗口推理”的完整工程含义。

---

## 7. UNETR++ 的核心代码剖析

这一节分成两部分：

1. 先讲论文原始思路，对应 `synapse/` 代码。
2. 再讲你当前工作区 ACDC 分支额外加了什么。

---

## 7.1 先看原始 UNETR++ 的整体结构

建议看：

- `network_architecture/synapse/unetr_pp_synapse.py`
- `network_architecture/synapse/model_components.py`
- `network_architecture/synapse/transformerblock.py`

整体结构可以概括成：

1. 输入 3D patch
2. 先做分层下采样编码
3. 每个 stage 使用 `TransformerBlock`
4. 最深层特征 reshape 成 token 形式
5. 解码器逐层上采样并和 encoder skip 相加
6. 输出最终分割图

### 一个高层伪代码

```text
def forward(x):
    x4, hidden_states = encoder(x)

    shallow_conv = encoder1(x)

    enc1, enc2, enc3, enc4 = hidden_states

    dec4 = reshape_tokens_to_feature_map(enc4)
    dec3 = up_block(dec4, enc3)
    dec2 = up_block(dec3, enc2)
    dec1 = up_block(dec2, enc1)
    out  = final_up_block(dec1, shallow_conv)

    return segmentation_heads(out, dec1, dec2)
```

这里的重点不是 U 形结构本身，而是：

- encoder/decoder 中间的 `TransformerBlock`
- `TransformerBlock` 里的 `EPA`

---

## 7.2 编码器：分层 patch embedding + Transformer stage

### 文件

- `network_architecture/synapse/model_components.py`

### 关键模块

- `UnetrPPEncoder`

### 它的工作方式

编码器不是一上来就把整个 3D 体数据 flatten 成超长 token 序列，而是先分层下采样：

1. `stem_layer`
   - 用卷积 stride 把原始体数据变成低分辨率高通道特征
2. `downsample_layers`
   - 每一层再进一步下采样
3. `stages`
   - 每个分辨率 stage 里堆若干个 `TransformerBlock`

这种设计比“单尺度超长序列 Transformer”更适合 3D 医学图像，因为：

- 3D 序列太长，直接全局 self-attention 代价极高
- 分层结构更接近 U-Net，也更容易做多尺度分割

---

## 7.3 TransformerBlock：真正的核心在 EPA

### 文件

- `network_architecture/synapse/transformerblock.py`

### forward 主流程

原始 synapse 版本的 `TransformerBlock.forward()` 大致是：

```text
1. [B, C, H, W, D] -> [B, N, C]
2. 加位置编码
3. norm 后送入 EPA
4. 残差相加
5. reshape 回 [B, C, H, W, D]
6. 经过局部卷积残差块增强
7. 输出
```

也就是：

```text
tokens = flatten(x)
attn = tokens + gamma * EPA(LN(tokens))
feat = reshape(attn)
feat = feat + conv_path(feat)
```

这里可以看出一个很重要的思想：

- 注意力模块负责长程依赖
- 卷积模块负责局部细节

这也是很多医学影像 Transformer 的经典做法。

---

## 7.4 EPA：Efficient Paired Attention 到底是什么

### 文件

- `network_architecture/synapse/transformerblock.py`

### 为什么 EPA 重要

普通 self-attention 对 3D 医学图像很贵，因为 token 数 `N = H*W*D` 很大。

UNETR++ 的核心创新之一，就是用 `EPA` 去替代纯粹的全量 self-attention。

EPA 做了两件事：

1. 通道注意力 `Channel Attention`
2. 空间注意力 `Spatial Attention`

而且这两条分支共享 query 和 key 的映射权重，因此叫 paired attention。

### 代码视角下的核心步骤

在 `EPA.forward(x)` 中：

1. `qkvv = Linear(x)`  
   一次性生成 `q_shared, k_shared, v_CA, v_SA`
2. `q_shared` 和 `k_shared` 被两个分支共用
3. 通道分支直接计算通道关系
4. 空间分支先把空间维投影到较小的 `proj_size`
5. 两条分支输出分别投影到 `C/2`
6. 最后拼接回来恢复到 `C`

### 一个更容易理解的伪代码

```text
q, k, v_channel, v_spatial = shared_linear(x)

# 通道注意力分支
attn_channel = softmax(normalize(q) @ normalize(k)^T)
x_channel = attn_channel @ v_channel

# 空间注意力分支
k_proj = E(k)
v_proj = F(v_spatial)
attn_spatial = softmax(q^T @ k_proj)
x_spatial = attn_spatial @ v_proj^T

# 两路结果融合
out = concat(proj_spatial(x_spatial), proj_channel(x_channel))
```

### EPA 的直觉解释

你可以把它理解成：

- 一路问“哪些通道彼此相关”
- 一路问“哪些空间位置彼此相关”
- 但两路不是完全独立学，而是共享部分表示

这样比纯 self-attention 更省计算，也保留了较强的全局建模能力。

---

## 7.5 解码器：为什么 skip connection 是“相加”不是“拼接”

### 文件

- `network_architecture/synapse/model_components.py`

`UnetrUpBlock.forward(inp, skip)` 是：

```text
out = transp_conv(inp)
out = out + skip
out = decoder_block(out)
```

这里和很多经典 U-Net 的 `concat` 不同，用的是 `add`。

这样做的特点是：

- 节省显存
- 通道数更稳定
- 更像残差融合

然后 `decoder_block` 里继续放 `TransformerBlock` 或卷积块，让上采样后的多尺度特征继续交互。

---

## 7.6 Deep Supervision：为什么网络会输出多个结果

### 文件

- `network_architecture/synapse/unetr_pp_synapse.py`

如果 `do_ds=True`，网络会返回：

```python
[self.out1(out), self.out2(dec1), self.out3(dec2)]
```

也就是说：

- 最终输出一张高分辨率分割图
- 中间层还会额外输出两张辅助分割图

训练时这些辅助头一起监督，推理时通常只用最终输出。

所以你会看到 trainer 在验证和推理前有一个重要动作：

- `self.network.do_ds = False`

意思就是：

“训练时保留多头输出，推理时关闭，只保留最终头。”

---

## 7.7 你当前工作区的 ACDC 网络，相比原始 UNETR++ 又加了什么

建议看：

- `network_architecture/acdc/unetr_pp_acdc.py`
- `network_architecture/acdc/model_components.py`
- `network_architecture/acdc/transformerblock.py`

这里有三类额外改动。

### 第一类：边缘分支

`LaplacianEdgeExtractor3D`

作用是：

- 对输入 MRI 做固定 3D Laplacian 卷积
- 提取高频边缘信息
- 再通过 `1x1x1 conv + norm + 激活` 映射为软边缘特征

后面在解码阶段：

- `decoder4`
- `decoder3`

会把这些边缘特征和主干特征拼接融合。

这说明当前 ACDC 版本不只是“分割语义”，还显式强调边界信息。

### 第二类：TransformerBlock 被替换成了 Mamba + Conv + Fusion

`acdc/transformerblock.py` 里不再是原始 `EPA`，而是：

1. `MobileMambaBlockSeq`
2. 局部卷积分支 `conv51 / conv52`
3. `MambaScConvCrossAttnFusion`
4. `conv8` 残差投影

所以这条线实际上已经是：

“Mamba 风格序列建模 + 卷积局部增强 + 跨分支融合”

而不再是论文原始的 EPA block。

### 第三类：解码器支持边缘特征拼接

`UnetrUpBlock` 增加了：

- `edge_channels`
- `edge_fusion_conv`
- `forward(..., edge_feat=None)`

这表示解码器不仅融合 skip feature，还会融合外部边缘特征。

所以如果你以后问“为什么 ACDC 的代码和 synapse 的代码长得不一样”，答案就是：

- `synapse/` 更接近原始 UNETR++
- `acdc/` 是你当前工作区里的实验性增强版本

---

## 8. 训练、验证、推理、patch 等概念澄清

这一节专门解决“概念晕”的问题。

---

## 8.1 Training（训练）

训练就是：

- 用带标签的数据
- 前向计算预测
- 用 loss 衡量预测和真值差距
- 反向传播更新参数

在本项目里，训练主要发生在：

- `NetworkTrainer_acdc.run_training()`
- `NetworkTrainer_acdc.run_iteration(do_backprop=True)`

训练阶段最核心的特点是：

- 参数会更新
- 输入通常是随机采样 patch
- 会做大量数据增强

---

## 8.2 Validation（验证）

验证就是：

- 用带标签的数据
- 只做前向，不更新参数
- 观察模型在未参与当前训练的样本上的表现

在本项目里有两种验证：

1. 训练中 patch 级验证
2. 训练后整卷验证

它们都属于 validation，但目的不同：

- 前者用于监控训练
- 后者用于出正式结果

---

## 8.3 Inference（推理）

推理就是：

- 用训练好的模型
- 对没有标签的新图像做预测
- 输出分割结果

在本项目里，推理相关主入口是：

- `unetr_pp/inference/predict_simple.py`
- `unetr_pp/inference/predict.py`

以及 trainer 里的：

- `preprocess_predict_nifti(...)`
- `predict_preprocessed_data_return_seg_and_softmax(...)`

推理和验证的区别在于：

- 推理通常没有 GT，不能算 Dice
- 验证有 GT，所以能计算指标

从工程路径上看，两者的预测流程很像，差别主要在“是否有参考标签做评估”。

---

## 8.4 Patch

patch 就是：

- 从完整 3D 体数据里切出来的一个小立方体或小长方体块

例如 ACDC 当前设置下：

- `patch_size = [16, 160, 160]`

表示网络一次看到的是：

- 16 层切片厚度
- 每层 160 x 160 的平面区域

为什么不整卷训练：

- 显存不够
- 训练效率低
- 类别不平衡更严重

所以 patch 是 3D 医学图像训练里非常常见的基本单位。

---

## 8.5 Fold

fold 指交叉验证中的一折。

`do_split()` 用 5 折 KFold，把数据拆成 5 份。

训练 `fold_0` 时：

- 4 份做训练
- 1 份做验证

如果你完整跑 `fold_0` 到 `fold_4`，就可以得到更稳健的交叉验证结果。

你现在脚本里只跑 `fold_0`，所以只是一个折的结果。

---

## 8.6 Stage

stage 是 nnU-Net 风格框架里的概念。

如果数据特别大，有时会做：

- 低分辨率 stage
- 高分辨率 stage

形成级联。

ACDC 当前通常只有一个 stage，所以 trainer 里最终使用的是：

- `stage = 0`

---

## 8.7 Plans 文件

plans 文件可以看成“实验蓝图”。

它记录了：

- patch size
- batch size
- spacing
- pooling 策略
- 归一化方案
- 数据标识名

没有它，trainer 不知道该怎样解释预处理数据，也不知道网络输入尺度该是多少。

---

## 8.8 Online Evaluation

`run_online_evaluation()` 不是最终论文评估，而是训练期间快速估计验证效果的机制。

它是在 patch 级输出上统计：

- TP
- FP
- FN

然后估算平均 Dice。

这个值主要服务于：

- 训练过程监控
- `model_best` 选择

---

## 8.9 Postprocessing（后处理）

在医学分割里，模型有时会预测出一些很小的离散噪点。

所以训练后会做连通域后处理：

- 对某些类别只保留最大连通域

关键文件：

- `postprocessing/connected_components.py`
- `Trainer_acdc.validate()` 里的 `determine_postprocessing(...)`

它会生成：

- `postprocessing.json`

以后推理时也可以复用这份规则。

---

## 9. 结果文件到底在哪里，分别代表什么

以当前 ACDC 输出目录为例：

- `output_acdc/unetr_pp/3d_fullres/Task001_ACDC/unetr_pp_trainer_acdc__unetr_pp_Plansv2.1/fold_0/`

里面常见文件含义如下：

- `model_final_checkpoint.model`
  最后一轮训练结束后的权重。
- `model_final_checkpoint.model.pkl`
  对应 checkpoint 的元信息，包含 trainer 初始化参数、plans 等。
- `model_best.model`
  训练过程中验证指标最优时的权重。如果没有触发，就可能不存在。
- `progress.png`
  训练/验证 loss 曲线。
- `debug.json`
  当前训练配置的完整快照。
- `training_log_*.txt`
  训练日志。
- `validation_raw/`
  原始整卷验证结果 NIfTI。
- `validation_raw/summary.json`
  验证集指标汇总。
- `validation_raw_postprocessed/`
  后处理后的验证结果。
- `postprocessing.json`
  后处理规则。

---

## 10. 你该怎么学习这个项目，才不会越看越乱

下面给你一个非常实用的学习顺序。

### 第 1 步：先跑通“入口到结果”的主链

按这个顺序看：

1. `training_scripts/run_training_acdc.sh`
2. `unetr_pp/run/run_training.py`
3. `unetr_pp/run/default_configuration.py`
4. `unetr_pp/training/network_training/unetr_pp_trainer_acdc.py`
5. `unetr_pp/training/network_training/Trainer_acdc.py`
6. `unetr_pp/training/network_training/network_trainer_acdc.py`

目标不是记住细节，而是回答：

- 训练是从哪里开始的？
- trainer 是怎么实例化的？
- 训练循环在哪？
- validate 在哪？

### 第 2 步：专门读“数据怎么进来”

按这个顺序看：

1. `unetr_pp/preprocessing/cropping.py`
2. `unetr_pp/preprocessing/preprocessing.py`
3. `unetr_pp/experiment_planning/DatasetAnalyzer.py`
4. `unetr_pp/training/dataloading/dataset_loading.py`
5. `unetr_pp/training/data_augmentation/default_data_augmentation.py`
6. `unetr_pp/training/data_augmentation/data_augmentation_moreDA.py`

目标是回答：

- 原始 NIfTI 怎样变成 `npz+pkl`？
- `class_locations` 为什么存在？
- patch 是怎么采样出来的？
- 数据增强在哪里做？

### 第 3 步：再读模型

如果你想先理解论文原始 UNETR++：

1. `network_architecture/synapse/unetr_pp_synapse.py`
2. `network_architecture/synapse/model_components.py`
3. `network_architecture/synapse/transformerblock.py`

如果你想理解当前工作区实际 ACDC 版本：

1. `network_architecture/acdc/unetr_pp_acdc.py`
2. `network_architecture/acdc/model_components.py`
3. `network_architecture/acdc/transformerblock.py`

### 第 4 步：最后读推理和评估

按这个顺序看：

1. `unetr_pp/network_architecture/neural_network.py`
2. `unetr_pp/inference/predict.py`
3. `unetr_pp/inference/segmentation_export.py`
4. `unetr_pp/evaluation/evaluator.py`
5. `unetr_pp/evaluation/metrics.py`

目标是回答：

- 滑动窗口到底怎么拼回整卷？
- 结果怎么恢复回原始 NIfTI 空间？
- `summary.json` 的 Dice/HD95 怎么算出来？

---

## 11. 当前工作区里值得你特别注意的几个“坑”

### 11.1 ACDC trainer 被改成了测试配置

`unetr_pp_trainer_acdc.py` 里：

- `self.max_num_epochs = 5`

这不是正式训练配置。

如果你以后复现实验结果，这里必须重新确认。

### 11.2 ACDC 模型不是论文最原始版本

当前 ACDC 版本加入了：

- Laplacian 边缘分支
- MobileMamba
- MambaScConv fusion

所以它更像“个人实验增强版”。

### 11.3 评估脚本里有明显参数命名疑点

`evaluation_scripts/run_evaluation_acdc.sh` 以及 synapse/lung 对应脚本里使用了：

```bash
-val
```

但 `run_training.py` 里真正注册的参数是：

- `-pred_pp`
- `--validation_only`

所以这些评估脚本看起来存在参数不匹配风险。  
如果你以后真正运行，建议优先检查这一点。

### 11.4 当前数据目录混用了 `Task01` 与 `Task001`

这是兼容历史目录结构造成的。

只要你知道：

- `Task01_ACDC` 更像外层数据目录
- `Task001_ACDC` 才是框架真正按 task name 识别的任务目录

就不会再被路径绕晕。

---

## 12. 一句话总结整个 ACDC 流程

如果把 ACDC 任务压缩成一句话：

`原始 MRI/标签 NIfTI -> 非零区域裁剪 -> 统计数据集特征 -> 生成 plans -> 重采样和归一化为 stage0 数据 -> KFold 划分 train/val -> DataLoader3D 随机采样 patch -> 强数据增强 -> UNETR++ 前向 + deep supervision loss -> patch 级在线验证 -> 保存 checkpoint -> 训练后整卷滑窗验证 -> 导出 NIfTI -> aggregate_scores 生成 summary.json -> 可选连通域后处理`

你以后再看任何一个文件，都要先问自己：

“它在这条总链上处于哪个位置？”

只要这个问题清楚，庞大的工程就不会再是散的。

---

## 13. 附录：按目录索引各个 `.py` 文件的作用

这一节不做深度解析，只做“查字典式索引”。当你以后看到某个文件名时，可以快速定位它大致负责什么。

---

## 13.1 根目录脚本

- `格式转化.py`：个人辅助数据格式转换脚本，不属于主训练框架主链。

---

## 13.2 `training_scripts/`

- `run_training_acdc.sh`：启动 ACDC 训练。
- `run_training_synapse.sh`：启动 Synapse 训练。
- `run_training_lung.sh`：启动 Lung 训练。
- `run_training_tumor.sh`：启动 BRaTs/Tumor 训练。

---

## 13.3 `evaluation_scripts/`

- `run_evaluation_acdc.sh`：调用 ACDC 验证流程，脚本参数需要额外核对。
- `run_evaluation_synapse.sh`：调用 Synapse 验证流程，脚本参数需要额外核对。
- `run_evaluation_lung.sh`：调用 Lung 验证流程，脚本参数需要额外核对。
- `run_evaluation_tumor.sh`：Tumor 的推理与评估脚本。

---

## 13.4 `unetr_pp/run/`

- `run_training.py`：训练总入口。
- `default_configuration.py`：根据任务和网络类型选择 plans、stage、trainer、输出目录。
- `__init__.py`：包初始化。

---

## 13.5 `unetr_pp/`

- `paths.py`：读取环境变量，定义 raw/preprocessed/results 目录。
- `configuration.py`：线程数、各向异性阈值等全局配置。
- `inference_acdc.py`：ACDC 专用离线评估脚本。
- `inference_synapse.py`：Synapse 专用离线评估脚本。
- `inference_tumor.py`：Tumor 专用离线评估脚本。
- `__init__.py`：包初始化。

---

## 13.6 `unetr_pp/training/network_training/`

- `network_trainer_acdc.py`：ACDC 训练循环抽象基类。
- `network_trainer_synapse.py`：Synapse 训练循环抽象基类。
- `network_trainer_lung.py`：Lung 训练循环抽象基类。
- `network_trainer_tumor.py`：Tumor 训练循环抽象基类。
- `Trainer_acdc.py`：ACDC 通用 trainer，负责 split、验证、评估、保存 `.pkl`。
- `Trainer_synapse.py`：Synapse 通用 trainer。
- `Trainer_lung.py`：Lung 通用 trainer。
- `Trainer_tumor.py`：Tumor 通用 trainer。
- `unetr_pp_trainer_acdc.py`：真正使用 UNETR++ ACDC 网络的 trainer。
- `unetr_pp_trainer_synapse.py`：真正使用 UNETR++ Synapse 网络的 trainer。
- `unetr_pp_trainer_lung.py`：真正使用 UNETR++ Lung 网络的 trainer。
- `unetr_pp_trainer_tumor.py`：真正使用 UNETR++ Tumor 网络的 trainer。

---

## 13.7 `unetr_pp/training/dataloading/`

- `dataset_loading.py`：加载预处理数据索引、解压 `npz->npy`、定义 `DataLoader3D/2D`、patch 采样逻辑。
- `__init__.py`：包初始化。

---

## 13.8 `unetr_pp/training/data_augmentation/`

- `default_data_augmentation.py`：基础数据增强配置与 pipeline。
- `data_augmentation_moreDA.py`：更强的数据增强 pipeline。
- `data_augmentation_noDA.py`：几乎不做增强的版本。
- `data_augmentation_insaneDA.py`：更激进的数据增强配置。
- `data_augmentation_insaneDA2.py`：另一版激进增强。
- `downsampling.py`：deep supervision 时对标签做多尺度下采样。
- `pyramid_augmentations.py`：级联分割特有的一些增强。
- `custom_transforms.py`：自定义 transform，例如 `MaskTransform`、2D/3D 变换。
- `__init__.py`：包初始化。

---

## 13.9 `unetr_pp/training/loss_functions/`

- `dice_loss.py`：Dice、GDL、CE+Dice 等核心 loss。
- `deep_supervision.py`：多输出 loss 加权封装。
- `crossentropy.py`：稳健版交叉熵。
- `TopK_loss.py`：Top-K loss。
- `__init__.py`：包初始化。

---

## 13.10 `unetr_pp/training/learning_rate/`

- `poly_lr.py`：poly 学习率策略。

---

## 13.11 `unetr_pp/training/optimizer/`

- `ranger.py`：Ranger 优化器实现。

---

## 13.12 `unetr_pp/training/cascade_stuff/`

- `predict_next_stage.py`：级联模型中把上一阶段输出预测到下一阶段。
- `__init__.py`：包初始化。

---

## 13.13 `unetr_pp/training/`

- `model_restore.py`：从 checkpoint 恢复 trainer 和网络；推理时很关键。
- `__init__.py`：包初始化。

---

## 13.14 `unetr_pp/network_architecture/` 核心公共模块

- `neural_network.py`：推理核心基类，含滑动窗口与 TTA 实现。
- `dynunet_block.py`：卷积块、残差块、上采样块、输出头。
- `generic_UNet.py`：通用 U-Net，实现很多 planner 和 trainer 依赖的基准网络逻辑。
- `layers.py`：自定义层，如 `LayerNorm`。
- `initialization.py`：权重初始化。
- `mobile_mamba_block.py`：MobileMamba 模块。
- `mamba_scconv_fusion.py`：Mamba 与 ScConv 跨分支融合模块。
- `ScConv.py`：ScConv 模块。
- `README.md`：网络结构目录说明。
- `__init__.py`：包初始化。

---

## 13.15 `unetr_pp/network_architecture/synapse/`

- `unetr_pp_synapse.py`：Synapse 版 UNETR++ 主网络。
- `model_components.py`：Synapse 版编码器/解码器组件。
- `transformerblock.py`：Synapse 版 `TransformerBlock` 与原始 `EPA`。
- `__init__.py`：包初始化。

---

## 13.16 `unetr_pp/network_architecture/acdc/`

- `unetr_pp_acdc.py`：当前工作区 ACDC 版 UNETR++ 主网络，含边缘分支。
- `model_components.py`：ACDC 版编码器/解码器组件，支持边缘特征融合。
- `transformerblock.py`：ACDC 版 TransformerBlock，融合 MobileMamba 与 ScConv。
- `mobile_mamba.py`：ACDC 分支单独保留的 Mamba 相关实现。
- `scconv.py`：ACDC 分支单独保留的 ScConv 相关实现。
- `__init__.py`：包初始化。

---

## 13.17 `unetr_pp/network_architecture/lung/`

- `unetr_pp_lung.py`：Lung 版 UNETR++ 主网络。
- `model_components.py`：Lung 版编码器/解码器组件。
- `transformerblock.py`：Lung 版 TransformerBlock。
- `__init__.py`：包初始化。

---

## 13.18 `unetr_pp/network_architecture/tumor/`

- `unetr_pp_tumor.py`：Tumor 版 UNETR++ 主网络。
- `model_components.py`：Tumor 版编码器/解码器组件。
- `transformerblock.py`：Tumor 版 TransformerBlock。
- `__init__.py`：包初始化。

---

## 13.19 `unetr_pp/preprocessing/`

- `preprocessing.py`：重采样、归一化、预处理主逻辑。
- `cropping.py`：非零区域裁剪。
- `sanity_checks.py`：数据集完整性检查。
- `custom_preprocessors/preprocessor_scale_RGB_to_0_1.py`：RGB 数据的特殊预处理。

---

## 13.20 `unetr_pp/experiment_planning/`

- `nnFormer_plan_and_preprocess.py`：计划与预处理总入口。
- `DatasetAnalyzer.py`：数据集统计分析。
- `utils.py`：split、crop、plan_and_preprocess 辅助函数。
- `common_utils.py`：规划过程中常用公共函数。
- `summarize_plans.py`：打印 plans 摘要。
- `change_batch_size.py`：修改 batch size 的辅助脚本。
- `nnFormer_convert_decathlon_task.py`：Decathlon 任务转换脚本。
- `experiment_planner_baseline_3DUNet.py`：3D planner 基类。
- `experiment_planner_baseline_3DUNet_v21.py`：3D planner v21。
- `experiment_planner_baseline_2DUNet.py`：2D planner 基类。
- `experiment_planner_baseline_2DUNet_v21.py`：2D planner v21。
- `__init__.py`：包初始化。

---

## 13.21 `unetr_pp/experiment_planning/alternative_experiment_planning/`

- `experiment_planner_baseline_3DUNet_v21_11GB.py`：针对 11GB 显存的 3D planner 变体。
- `experiment_planner_baseline_3DUNet_v21_16GB.py`：针对 16GB 显存的 3D planner 变体。
- `experiment_planner_baseline_3DUNet_v21_32GB.py`：针对 32GB 显存的 3D planner 变体。
- `experiment_planner_baseline_3DUNet_v21_3convperstage.py`：每 stage 三层卷积的变体。
- `experiment_planner_residual_3DUNet_v21.py`：残差 3D U-Net planner 变体。
- `experiment_planner_baseline_3DUNet_v22.py`：3D planner v22 变体。
- `experiment_planner_baseline_3DUNet_v23.py`：3D planner v23 变体。

---

## 13.22 `unetr_pp/experiment_planning/alternative_experiment_planning/normalization/`

- `experiment_planner_2DUNet_v21_RGB_scaleto_0_1.py`：RGB 2D 数据归一化到 0-1。
- `experiment_planner_3DUNet_CT2.py`：CT2 归一化策略。
- `experiment_planner_3DUNet_nonCT.py`：非 CT 归一化策略。

---

## 13.23 `unetr_pp/experiment_planning/alternative_experiment_planning/patch_size/`

- `experiment_planner_3DUNet_isotropic_in_voxels.py`：按 voxel 各向同性决定 patch。
- `experiment_planner_3DUNet_isotropic_in_mm.py`：按物理毫米尺度各向同性决定 patch。

---

## 13.24 `unetr_pp/experiment_planning/alternative_experiment_planning/target_spacing/`

- `experiment_planner_baseline_3DUNet_targetSpacingForAnisoAxis.py`：针对各向异性轴的 spacing 变体。
- `experiment_planner_baseline_3DUNet_v21_noResampling.py`：不重采样变体。
- `experiment_planner_baseline_3DUNet_v21_customTargetSpacing_2x2x2.py`：固定目标 spacing 为 2x2x2。

---

## 13.25 `unetr_pp/experiment_planning/alternative_experiment_planning/pooling_and_convs/`

- `experiment_planner_baseline_3DUNet_poolBasedOnSpacing.py`：基于 spacing 的 pooling 变体。
- `experiment_planner_baseline_3DUNet_allConv3x3.py`：全部使用 3x3 卷积的变体。

---

## 13.26 `unetr_pp/inference/`

- `predict.py`：整套推理主逻辑，支持文件夹批量预测。
- `predict_simple.py`：更常用的推理 CLI 封装。
- `segmentation_export.py`：把 softmax 或 seg 恢复到原始空间并写成 NIfTI。
- `__init__.py`：包初始化。

---

## 13.27 `unetr_pp/evaluation/`

- `evaluator.py`：评估器主逻辑。
- `metrics.py`：Dice、Jaccard、Precision、Recall、HD95 等指标。
- `surface_dice.py`：Surface Dice 相关实现。
- `region_based_evaluation.py`：基于区域的评估。
- `collect_results_files.py`：收集结果文件。
- `add_mean_dice_to_json.py`：向评估 JSON 中追加 mean Dice。
- `add_dummy_task_with_mean_over_all_tasks.py`：补充伪任务汇总平均结果。
- `__init__.py`：包初始化。

---

## 13.28 `unetr_pp/evaluation/model_selection/`

- `rank_candidates.py`：候选模型排序。
- `rank_candidates_cascade.py`：级联模型候选排序。
- `rank_candidates_StructSeg.py`：StructSeg 特化排序。
- `ensemble.py`：模型集成。
- `summarize_results_in_one_json.py`：汇总多个结果 JSON。
- `summarize_results_with_plans.py`：结合 plans 汇总结果。
- `collect_all_fold0_results_and_summarize_in_one_csv.py`：收集所有 fold0 结果并写 CSV。
- `figure_out_what_to_submit.py`：帮助决定提交哪些结果。
- `__init__.py`：包初始化。

---

## 13.29 `unetr_pp/postprocessing/`

- `connected_components.py`：连通域后处理核心实现。
- `consolidate_postprocessing.py`：整合后处理规则。
- `consolidate_postprocessing_simple.py`：简化版后处理整合。
- `consolidate_all_for_paper.py`：为论文结果汇总全部后处理。

---

## 13.30 `unetr_pp/utilities/`

- `task_name_id_conversion.py`：任务名和任务 ID 转换。
- `to_torch.py`：numpy/tensor/GPU 转换。
- `tensor_utilities.py`：tensor 工具函数。
- `nd_softmax.py`：softmax 辅助函数。
- `one_hot_encoding.py`：one-hot 编码。
- `folder_names.py`：文件夹命名辅助。
- `file_endings.py`：后缀处理辅助。
- `file_conversions.py`：文件格式转换辅助。
- `sitk_stuff.py`：SimpleITK 辅助工具。
- `overlay_plots.py`：可视化叠加图。
- `distributed.py`：分布式训练相关辅助。
- `random_stuff.py`：随机数相关辅助。
- `recursive_delete_npz.py`：递归删除 npz。
- `recursive_rename_taskXX_to_taskXXX.py`：任务名批量重命名。
- `__init__.py`：包初始化。

---

## 14. 最后的建议

如果你只打算认真弄懂一次这个项目，我建议你做一件非常有效的事情：

拿着这份文档，自己再画一张只包含 10 个框的流程图：

`run_training.sh -> run_training.py -> get_default_configuration -> trainer.initialize -> load_dataset/do_split -> DataLoader3D -> augmentation -> UNETR_PP.forward -> loss/backward -> validate/predict/export`

只要这 10 个框你能闭眼讲出来，这个项目你就已经不再“晕”了。

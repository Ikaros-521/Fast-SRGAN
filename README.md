# Fast-SRGAN
本仓库的目标是实现低分辨率视频的实时超分辨率放大。目前的设计遵循 [SR-GAN](https://arxiv.org/pdf/1609.04802.pdf) 架构。为了提高速度，上采样通过像素混洗（pixel shuffle）完成。

训练设置如下图所示：

<p align="center">
  <img src="https://user-images.githubusercontent.com/4294680/67164120-22157480-f377-11e9-87c1-5b6acace0e47.png">
</p>

# 速度基准测试
以下运行时间/帧率是通过对800帧的运行时间取平均值获得的。在MPS（MacBook M1 Pro GPU）上测量。

| 输入图像尺寸  |      输出尺寸      | 时间 (s)  | 帧率 |
|   ---------   |:-----------------:|:---------:|:---:|
|     90x160    |    360x640 (360p)  |   0.01    | 82  |
|     180x320   |    720x1080 (720p) |   0.04    | 27  |

我们可以看到可以以约30fps的速度上采样到720p。

# 环境要求
此项目在Python 3.10上测试。

## 图像推理
要安装图像推理所需的包，请使用提供的Pipfile：
```bash
pip install pipenv --upgrade
pipenv install --system --deploy
```

## 视频推理
要使用视频推理功能，需要安装额外的依赖：
```bash
# 安装视频推理依赖
pip install -r requirements_video.txt

# 或者手动安装
pip install opencv-python psutil
```

# 预训练模型
在'models'目录中提供了在DIV2k数据集上预训练的生成器模型。它使用8个残差块，生成器的每一层都有64个滤波器。

要在您自己的图像上试用提供的预训练模型，请运行以下命令：

```bash
python inference.py --image_dir 'path/to/your/image/directory' --output_dir 'path/to/save/super/resolution/images'
```

# 视频超分辨率推理

本项目现在支持视频文件的超分辨率处理！我们提供了两个视频推理脚本：

## 基础视频推理

使用 `video_inference.py` 进行基础视频处理：

```bash
python video_inference.py --input_video 'path/to/input/video.mp4' --output_video 'path/to/output/video.mp4'
```

## 高级视频推理

使用 `video_inference_advanced.py` 进行高级视频处理，支持更多功能：

```bash
# 基础使用
python video_inference_advanced.py --input_video 'input.mp4' --output_video 'output.mp4'

# 自定义批处理大小和内存限制
python video_inference_advanced.py --input_video 'input.mp4' --output_video 'output.mp4' --batch_size 4 --max_memory 6

# 处理视频片段
python video_inference_advanced.py --input_video 'input.mp4' --output_video 'output.mp4' --start_frame 100 --end_frame 500

# 指定设备
python video_inference_advanced.py --input_video 'input.mp4' --output_video 'output.mp4' --device cuda
```

### 高级功能特性

- **智能内存管理**: 自动监控内存使用，防止内存溢出
- **批处理优化**: 支持批量处理帧以提高效率
- **视频片段处理**: 可以处理视频的特定片段
- **实时进度显示**: 显示处理进度、FPS和内存使用情况
- **多设备支持**: 自动选择最佳设备（CUDA/MPS/CPU）
- **错误处理**: 完善的错误处理和资源清理

### 参数说明

- `--input_video`: 输入视频文件路径
- `--output_video`: 输出视频文件路径
- `--batch_size`: 批处理大小（默认: 1，建议根据GPU内存调整）
- `--max_memory`: 最大内存使用量，单位GB（默认: 8）
- `--start_frame`: 起始帧（默认: 0）
- `--end_frame`: 结束帧（默认: 全部）
- `--device`: 指定设备（cuda/cpu/mps）
- `--config`: 配置文件路径（默认: configs/config.yaml）
- `--model`: 模型文件路径（默认: models/model.pt）

### 性能建议

- **GPU内存充足**: 可以增加 `batch_size` 以提高处理速度
- **GPU内存不足**: 减少 `batch_size` 或降低 `max_memory` 限制
- **长视频处理**: 建议使用视频片段功能分段处理
- **实时处理**: 对于实时应用，建议使用较小的批处理大小

## 快速开始

运行示例脚本快速体验视频超分辨率功能：

```bash
python video_example.py
```

这个脚本会引导您完成各种视频推理示例。

# 训练
要训练模型，只需在`configs/config.yaml`文件夹中编辑配置文件，然后启动训练：
```bash
python train.py
```

您也可以从命令行更改配置参数。以下命令将以32的`batch_size`、12个残差块的生成器以及图像目录路径`/path/to/image/dataset`运行训练：
```
python train.py data.image_dir="/path/to/image/dataset" training.batch_size=32 generator.n_layers=12

```
这由`hydra`提供支持，这意味着配置中的所有参数都可以通过CLI进行编辑。

模型检查点和训练摘要保存在tensorboard中。要监控训练进度，请将tensorboard指向训练开始时创建的`outputs`目录。

# 样本结果
以下是提供的训练模型的一些结果。左侧显示低分辨率图像，经过4倍双三次上采样。中间是模型的输出。右侧是实际的高分辨率图像。

<p align="center">
  <b>以下显示了通过双三次插值4倍上采样的图像、本仓库的预训练模型以及原始高分辨率图像作为对比</b>
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/95b6f8e4-f6c0-403b-854e-78c5589fbec6g"> 
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/d57abe02-46d8-48ce-bd1f-3cd06a9a66087">
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/67472974-56a5-4505-abaa-5e1c86467da1">
  <img src="https://github.com/HasnainRaz/Fast-SRGAN/assets/4294680/0d16647e-75ea-4150-bba0-2ea70ba05ca0">
</p>

# 贡献
如果您有改进模型性能、添加指标或任何其他更改的想法，请提交拉取请求或开启问题。我很乐意接受任何贡献。


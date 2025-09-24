import os
# 禁用 Triton 以避免兼容性问题
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cv2
import torch
import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
import gc
import psutil
import time

from model import Generator


class VideoSuperResolution:
    """视频超分辨率处理类"""
    
    def __init__(self, config_path="configs/config.yaml", model_path="models/model.pt", device=None):
        self.device = device or self._get_device()
        self.model = self._load_model(config_path, model_path)
        self.scale_factor = 4  # 从配置文件读取
        
    def _get_device(self):
        """自动选择最佳设备"""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"使用CUDA设备: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("使用MPS设备 (Apple Silicon)")
        else:
            device = "cpu"
            print("使用CPU设备")
        return device
    
    def _load_model(self, config_path, model_path):
        """加载预训练的SRGAN模型"""
        print("加载模型...")
        config = OmegaConf.load(config_path)
        model = Generator(config.generator)
        
        # 加载权重
        weights = torch.load(model_path, map_location="cpu")
        new_weights = {}
        for k, v in weights.items():
            new_weights[k.replace("_orig_mod.", "")] = v
        model.load_state_dict(new_weights)
        
        model.to(self.device)
        model.eval()
        
        # 如果使用CUDA，启用优化
        if self.device == "cuda":
            model = torch.compile(model)
            torch.backends.cudnn.benchmark = True
        
        print("模型加载完成！")
        return model
    
    def _preprocess_frame(self, frame):
        """预处理视频帧"""
        # 转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)
        # 转换为numpy数组并归一化
        frame_array = np.array(pil_image)
        frame_tensor = (torch.from_numpy(frame_array) / 127.5) - 1.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(dim=0)
        return frame_tensor
    
    def _postprocess_frame(self, sr_tensor):
        """后处理超分辨率帧"""
        # 将张量从 [-1, 1] 范围转换到 [0, 1]
        sr_tensor = (sr_tensor + 1.0) / 2.0
        
        # 处理不同维度的张量
        if sr_tensor.dim() == 4:  # [batch, channels, height, width]
            # 移除批次维度并重新排列
            sr_tensor = sr_tensor.squeeze(0)  # 移除批次维度
            sr_tensor = sr_tensor.permute(1, 2, 0)  # [height, width, channels]
        elif sr_tensor.dim() == 3:  # [channels, height, width]
            sr_tensor = sr_tensor.permute(1, 2, 0)  # [height, width, channels]
        else:
            raise ValueError(f"不支持的张量维度: {sr_tensor.dim()}")
        
        # 转换为numpy数组
        sr_array = (sr_tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
        return sr_array
    
    def _get_memory_usage(self):
        """获取当前内存使用情况"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            return psutil.Process().memory_info().rss / 1024**3  # GB
    
    def process_video(self, input_path, output_path, batch_size=1, max_memory_gb=8):
        """处理视频文件"""
        print(f"开始处理视频: {input_path}")
        
        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"输入视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  输出分辨率: {width*self.scale_factor}x{height*self.scale_factor}")
        print(f"  批处理大小: {batch_size}")
        
        # 动态调整批处理大小基于内存使用
        initial_memory = self._get_memory_usage()
        print(f"初始内存使用: {initial_memory:.2f} GB")
        
        # 设置输出视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*self.scale_factor, height*self.scale_factor))
        
        # 存储帧的列表
        frames_batch = []
        frame_count = 0
        start_time = time.time()
        
        try:
            with tqdm(total=total_frames, desc="处理视频帧", unit="帧") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 预处理帧
                    frame_tensor = self._preprocess_frame(frame)
                    frames_batch.append(frame_tensor)
                    
                    # 检查内存使用情况
                    current_memory = self._get_memory_usage()
                    if current_memory > max_memory_gb:
                        print(f"\n内存使用过高 ({current_memory:.2f} GB)，强制处理当前批次")
                        batch_size = max(1, batch_size // 2)
                    
                    # 当达到批次大小或最后一帧时，处理批次
                    if len(frames_batch) >= batch_size or frame_count == total_frames - 1:
                        # 合并批次
                        batch_tensor = torch.cat(frames_batch, dim=0).to(self.device)
                        
                        # 模型推理
                        with torch.no_grad():
                            sr_batch = self.model(batch_tensor).cpu()
                        
                        # 后处理并写入输出视频
                        for i in range(sr_batch.shape[0]):
                            # 确保单个帧张量有正确的维度
                            single_frame = sr_batch[i]  # 这应该是 [channels, height, width]
                            sr_frame = self._postprocess_frame(single_frame)
                            # 转换回BGR格式用于OpenCV
                            sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
                            out.write(sr_frame_bgr)
                        
                        # 清理内存
                        del batch_tensor, sr_batch
                        frames_batch = []
                        
                        # 定期清理GPU缓存
                        if self.device == "cuda" and frame_count % 100 == 0:
                            torch.cuda.empty_cache()
                            gc.collect()
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # 更新进度信息
                    if frame_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps_processed = frame_count / elapsed_time
                        pbar.set_postfix({
                            'FPS': f'{fps_processed:.1f}',
                            'Memory': f'{self._get_memory_usage():.1f}GB'
                        })
        
        finally:
            # 释放资源
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # 清理内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time
        print(f"\n视频处理完成！")
        print(f"输出保存到: {output_path}")
        print(f"总处理时间: {total_time:.2f} 秒")
        print(f"平均处理速度: {avg_fps:.2f} FPS")
    
    def process_video_segment(self, input_path, output_path, start_frame=0, end_frame=None, batch_size=1):
        """处理视频片段"""
        print(f"处理视频片段: 帧 {start_frame} 到 {end_frame or '结束'}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
        
        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 设置输出视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*self.scale_factor, height*self.scale_factor))
        
        frames_batch = []
        frame_count = start_frame
        
        try:
            with tqdm(total=end_frame-start_frame, desc="处理视频片段", unit="帧") as pbar:
                while frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_tensor = self._preprocess_frame(frame)
                    frames_batch.append(frame_tensor)
                    
                    if len(frames_batch) >= batch_size or frame_count == end_frame - 1:
                        batch_tensor = torch.cat(frames_batch, dim=0).to(self.device)
                        
                        with torch.no_grad():
                            sr_batch = self.model(batch_tensor).cpu()
                        
                        for i in range(sr_batch.shape[0]):
                            sr_frame = self._postprocess_frame(sr_batch[i])
                            sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
                            out.write(sr_frame_bgr)
                        
                        frames_batch = []
                    
                    frame_count += 1
                    pbar.update(1)
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"视频片段处理完成！输出保存到: {output_path}")


def main():
    parser = ArgumentParser("视频超分辨率推理")
    parser.add_argument("--input_video", required=True, type=str, help="输入视频文件路径")
    parser.add_argument("--output_video", required=True, type=str, help="输出视频文件路径")
    parser.add_argument("--batch_size", default=1, type=int, help="批处理大小 (默认: 1)")
    parser.add_argument("--max_memory", default=8, type=float, help="最大内存使用量 (GB, 默认: 8)")
    parser.add_argument("--start_frame", default=0, type=int, help="起始帧 (默认: 0)")
    parser.add_argument("--end_frame", default=None, type=int, help="结束帧 (默认: 全部)")
    parser.add_argument("--config", default="configs/config.yaml", type=str, help="配置文件路径")
    parser.add_argument("--model", default="models/model.pt", type=str, help="模型文件路径")
    parser.add_argument("--device", default="cuda", type=str, help="指定设备 (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"输入视频文件不存在: {args.input_video}")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建视频超分辨率处理器
    vsr = VideoSuperResolution(args.config, args.model, args.device)
    
    # 处理视频
    try:
        if args.start_frame > 0 or args.end_frame is not None:
            vsr.process_video_segment(
                args.input_video, 
                args.output_video, 
                args.start_frame, 
                args.end_frame, 
                args.batch_size
            )
        else:
            vsr.process_video(
                args.input_video, 
                args.output_video, 
                args.batch_size, 
                args.max_memory
            )
    except Exception as e:
        print(f"处理视频时出错: {e}")
        raise


if __name__ == "__main__":
    main()

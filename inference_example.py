#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21点姿态估计模型推理示例
支持单张图片、视频和批量图片推理
"""

from mmpose.apis import MMPoseInferencer
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='21点姿态估计模型推理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单张图片推理
  python inference_example.py image.jpg
  
  # 视频推理
  python inference_example.py video.mp4
  
  # 批量图片推理
  python inference_example.py images/ --batch
  
  # 指定输出目录
  python inference_example.py image.jpg --output results/
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='输入文件路径（图片或视频）或目录路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel.py',
        help='模型配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_coco_AP_epoch_110.pth',
        help='权重文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='vis_results',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推理设备 (cuda:0, cuda:1, 或 cpu)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批量处理模式（输入为目录时）'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示结果（不保存）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 检查权重文件
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: 权重文件不存在: {args.checkpoint}")
        print("请先下载权重文件并放到 checkpoints/ 目录")
        return
    
    # 创建推理器
    print(f"正在加载模型: {args.checkpoint}")
    try:
        inferencer = MMPoseInferencer(
            pose2d=args.config,
            pose2d_weights=args.checkpoint,
            device=args.device
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 处理输入
    if input_path.is_file():
        # 单文件处理
        print(f"正在推理: {args.input}")
        try:
            # 确保输出目录存在
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.show:
                # 显示结果
                result_generator = inferencer(args.input, show=True)
                # 遍历生成器以处理所有帧
                for result in result_generator:
                    pass
                print("推理完成（显示模式）")
            else:
                # 保存结果 - 对于视频需要使用生成器并遍历所有结果
                result_generator = inferencer(args.input, vis_out_dir=str(output_dir))
                # 遍历生成器以处理所有帧（视频需要处理所有帧）
                frame_count = 0
                for result in result_generator:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"  已处理 {frame_count} 帧...")
                
                print(f"✓ 推理完成，共处理 {frame_count} 帧")
                print(f"✓ 结果已保存到: {args.output}")
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            
    elif input_path.is_dir() and args.batch:
        # 批量处理
        print(f"批量处理目录: {args.input}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("错误: 目录中没有找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 确保输出目录存在
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {img_path.name}")
            try:
                # 对于图片，也需要遍历生成器
                result_generator = inferencer(str(img_path), vis_out_dir=str(output_dir))
                for result in result_generator:
                    pass
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
        
        print(f"✓ 批量处理完成，结果保存在: {args.output}")
    else:
        print("错误: 输入路径无效或未使用 --batch 参数")
        print("提示: 对于目录输入，请使用 --batch 参数")


if __name__ == '__main__':
    main()

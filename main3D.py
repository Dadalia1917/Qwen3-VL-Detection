#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 基于Qwen3-VL的纸箱3D检测系统

import os
import json
import time
import math
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection_3d.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Carton3DDetectionSystem:
    def __init__(self, model_path="./models", datasets_dir="./datasets/test",
                 outputs_dir="./outputs/3D", device="auto", fov=60.0):
        self.model_path = model_path
        self.datasets_dir = Path(datasets_dir)
        self.outputs_dir = Path(outputs_dir)
        self.device = device
        self.default_fov = fov
        
        self.setup_output_directories()
        logger.info(f"正在加载模型: {model_path}")
        self.load_model()
        
    def setup_output_directories(self):
        subdirs = ['json', 'images', 'visualizations', 'raw', 'camera_params']
        for subdir in subdirs:
            (self.outputs_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {self.outputs_dir}")
        
    def load_model(self):
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info(f"模型加载成功: {self.model.device}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_image_files(self):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        if self.datasets_dir.exists():
            for root, dirs, files in os.walk(self.datasets_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(Path(root) / file)
        
        logger.info(f"找到 {len(image_files)} 张图片")
        return image_files
    
    def generate_camera_params(self, image_path):
        image = Image.open(image_path)
        w, h = image.size
        
        fx = round(w / (2 * np.tan(np.deg2rad(self.default_fov) / 2)), 2)
        fy = round(h / (2 * np.tan(np.deg2rad(self.default_fov) / 2)), 2)
        cx = round(w / 2, 2)
        cy = round(h / 2, 2)
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'fov': self.default_fov,
            'image_width': w, 'image_height': h
        }
    
    def detect_cartons_3d(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 限制图片尺寸
            max_size = 1024
            if max(original_size) > max_size:
                ratio = max_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            cam_params = self.generate_camera_params(image_path)
            
            # 使用简洁的提示词（参考官方示例）
            user_prompt = "Find all cartons and boxes in this image. For each box, provide its 3D bounding box. The output format required is JSON: [{\"bbox_3d\":[x, y, z, x_size, y_size, z_size, roll, pitch, yaw], \"label\":\"carton\"}]."
            
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image}
                ]}
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            detections = self.parse_3d_detections(output_text)
            
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'detection_type': '3D',
                'original_size': {'width': original_size[0], 'height': original_size[1]},
                'camera_params': cam_params,
                'raw_output': output_text,
                'detections': detections,
                'num_detections': len(detections),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"{image_path.name}: 检测到 {len(detections)} 个纸箱")
            return result
            
        except Exception as e:
            logger.error(f"处理失败 {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'detection_type': '3D',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def parse_3d_detections(self, output_text):
        detections = []
        try:
            import re
            
            # 清理文本
            cleaned_text = output_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.split('```json')[1].split('```')[0]
            elif '[' in cleaned_text and ']' in cleaned_text:
                start = cleaned_text.find('[')
                end = cleaned_text.rfind(']') + 1
                cleaned_text = cleaned_text[start:end]
            
            # 修复常见JSON格式问题
            cleaned_text = re.sub(r"'([^']*)':", r'"\1":', cleaned_text)
            cleaned_text = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned_text)
            
            data = json.loads(cleaned_text.strip())
            if isinstance(data, dict):
                data = [data]
            
            # 用于去重的集合
            seen_bboxes = set()
            
            for item in data:
                if 'bbox_3d' in item and len(item['bbox_3d']) == 9:
                    # 将bbox转为tuple用于去重判断
                    bbox_tuple = tuple(item['bbox_3d'])
                    
                    # 如果是完全相同的bbox就跳过（只保留第一个）
                    if bbox_tuple in seen_bboxes:
                        continue
                    
                    seen_bboxes.add(bbox_tuple)
                    
                    detection = {
                        'bbox_3d': item['bbox_3d'],
                        'label': item.get('label', 'carton'),
                        'confidence': item.get('confidence', 1.0)
                    }
                    detections.append(detection)
                    
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败: {output_text[:200]}")
        except Exception as e:
            logger.warning(f"解析失败: {e}")
        
        return detections
    
    def convert_3dbbox_to_2d(self, bbox_3d, cam_params):
        x, y, z, x_size, y_size, z_size, roll, pitch, yaw = bbox_3d
        pitch_deg = pitch * 180
        yaw_deg = yaw * 180
        roll_deg = roll * 180
        
        hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
        local_corners = [
            [ hx,  hy,  hz], [ hx,  hy, -hz],
            [ hx, -hy,  hz], [ hx, -hy, -hz],
            [-hx,  hy,  hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [-hx, -hy, -hz]
        ]
        
        def rotate_xyz(point, pitch_rad, yaw_rad, roll_rad):
            x0, y0, z0 = point
            x1 = x0
            y1 = y0 * math.cos(pitch_rad) - z0 * math.sin(pitch_rad)
            z1 = y0 * math.sin(pitch_rad) + z0 * math.cos(pitch_rad)
            x2 = x1 * math.cos(yaw_rad) + z1 * math.sin(yaw_rad)
            y2 = y1
            z2 = -x1 * math.sin(yaw_rad) + z1 * math.cos(yaw_rad)
            x3 = x2 * math.cos(roll_rad) - y2 * math.sin(roll_rad)
            y3 = x2 * math.sin(roll_rad) + y2 * math.cos(roll_rad)
            z3 = z2
            return [x3, y3, z3]
        
        img_corners = []
        for corner in local_corners:
            rotated = rotate_xyz(corner, np.deg2rad(pitch_deg), 
                               np.deg2rad(yaw_deg), np.deg2rad(roll_deg))
            X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
            if Z > 0:
                x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
                y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
                img_corners.append([x_2d, y_2d])
        
        return img_corners
    
    def save_results(self, result):
        if 'error' in result:
            return
        
        image_name = Path(result['image_name']).stem
        
        # 保存JSON
        json_path = self.outputs_dir / 'json' / f"{image_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存原始输出
        raw_path = self.outputs_dir / 'raw' / f"{image_name}_raw.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(result['raw_output'])
        
        # 保存相机参数
        cam_path = self.outputs_dir / 'camera_params' / f"{image_name}_camera.json"
        with open(cam_path, 'w', encoding='utf-8') as f:
            json.dump(result['camera_params'], f, indent=2)
        
        self.visualize_3d_detections(result)
    
    def visualize_3d_detections(self, result):
        try:
            image_cv = cv2.imread(result['image_path'])
            if image_cv is None:
                return
            
            edges = [
                [0,1], [2,3], [4,5], [6,7],
                [0,2], [1,3], [4,6], [5,7],
                [0,4], [1,5], [2,6], [3,7]
            ]
            
            for i, det in enumerate(result['detections']):
                bbox_2d = self.convert_3dbbox_to_2d(det['bbox_3d'], result['camera_params'])
                
                if len(bbox_2d) >= 8:
                    color = (np.random.randint(0, 255), 
                            np.random.randint(0, 255), 
                            np.random.randint(0, 255))
                    
                    for start, end in edges:
                        if start < len(bbox_2d) and end < len(bbox_2d):
                            pt1 = tuple([int(pt) for pt in bbox_2d[start]])
                            pt2 = tuple([int(pt) for pt in bbox_2d[end]])
                            cv2.line(image_cv, pt1, pt2, color, 2)
                    
                    if len(bbox_2d) > 0:
                        confidence = det.get('confidence', 1.0)
                        label = f"{det['label']} {confidence:.2f}"
                        pt = tuple([int(coord) for coord in bbox_2d[0]])
                        cv2.putText(image_cv, label, pt, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            image_name = Path(result['image_name']).stem
            vis_path = self.outputs_dir / 'images' / f"{image_name}_3d_detection.jpg"
            cv2.imwrite(str(vis_path), image_cv)
            
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)
            ax.axis('off')
            ax.set_title(f'3D: {result["image_name"]} ({result["num_detections"]})')
            
            vis_path_plt = self.outputs_dir / 'visualizations' / f"{image_name}_3d_vis.png"
            plt.savefig(vis_path_plt, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"可视化失败: {e}")
    
    def process_all_images(self):
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("没有找到图片")
            return
        
        logger.info(f"开始处理 {len(image_files)} 张图片...")
        
        results_summary = {
            'detection_type': '3D',
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'camera_fov': self.default_fov,
            'results': []
        }
        
        for image_path in tqdm(image_files, desc="3D检测"):
            try:
                result = self.detect_cartons_3d(image_path)
                self.save_results(result)
                
                if 'error' not in result:
                    results_summary['processed'] += 1
                    results_summary['total_detections'] += result['num_detections']
                else:
                    results_summary['failed'] += 1
                
                results_summary['results'].append({
                    'image_name': result['image_name'],
                    'num_detections': result.get('num_detections', 0),
                    'status': 'success' if 'error' not in result else 'failed'
                })
                
                # 每10张清理显存
                if (results_summary['processed'] + results_summary['failed']) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except KeyboardInterrupt:
                logger.warning("用户中断处理")
                break
            except Exception as e:
                logger.error(f"处理失败 {image_path}: {e}")
                results_summary['failed'] += 1
        
        summary_path = self.outputs_dir / 'detection_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"处理完成!")
        logger.info(f"总数: {results_summary['total_images']}")
        logger.info(f"成功: {results_summary['processed']}")
        logger.info(f"失败: {results_summary['failed']}")
        logger.info(f"检测总数: {results_summary['total_detections']}")
        logger.info(f"平均: {results_summary['total_detections']/max(results_summary['processed'],1):.2f}")
        logger.info(f"FOV: {self.default_fov}°")
        logger.info(f"输出: {self.outputs_dir}")
        logger.info(f"{'='*50}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="纸箱3D检测系统")
    parser.add_argument('--model', '-m', default='./models', help='模型路径')
    parser.add_argument('--datasets', '-d', default='./datasets', help='数据集目录')
    parser.add_argument('--outputs', '-o', default='./outputs/3D', help='输出目录')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='设备')
    parser.add_argument('--fov', type=float, default=60.0, help='相机FOV(度)')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("3D检测系统")
    logger.info(f"模型: {args.model}")
    logger.info(f"数据集: {args.datasets}")
    logger.info(f"输出: {args.outputs}")
    logger.info(f"设备: {args.device}")
    logger.info(f"FOV: {args.fov}°")
    logger.info("="*60)
    
    detector = Carton3DDetectionSystem(
        model_path=args.model,
        datasets_dir=args.datasets,
        outputs_dir=args.outputs,
        device=args.device,
        fov=args.fov
    )
    
    detector.process_all_images()


if __name__ == "__main__":
    main()

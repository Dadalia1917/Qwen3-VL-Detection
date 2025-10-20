#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 基于Qwen3-VL的纸箱2D检测系统

import os
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection_2d.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Carton2DDetectionSystem:
    def __init__(self, model_path="./models", datasets_dir="./datasets/test",
                 outputs_dir="./outputs/2D", device="auto"):
        self.model_path = model_path
        self.datasets_dir = Path(datasets_dir)
        self.outputs_dir = Path(outputs_dir)
        self.device = device
        
        self.setup_output_directories()
        logger.info(f"正在加载模型: {model_path}")
        self.load_model()
        
    def setup_output_directories(self):
        subdirs = ['json', 'images', 'labels', 'raw']
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
    
    def detect_cartons(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 限制图片尺寸
            max_size = 1024
            if max(original_size) > max_size:
                ratio = max_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 使用更详细的提示词
            system_prompt = "You are a helpful assistant specializing in object detection. You must provide confidence scores for each detection."
            user_prompt = "Please detect all cardboard boxes and cartons in this image. For each detection, provide bounding box coordinates and confidence score in JSON format: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'carton', 'confidence': 0.95}]. The confidence score should be between 0 and 1, representing how certain you are about the detection. Only detect clearly visible boxes."
            
            messages = [
                {"role": "system", "content": system_prompt},
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
            
            # 清理显存
            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            detections = self.parse_detections(output_text, original_size)
            
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'detection_type': '2D',
                'original_size': {'width': original_size[0], 'height': original_size[1]},
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
                'detection_type': '2D',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def parse_detections(self, output_text, image_size):
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
            
            data = json.loads(cleaned_text)
            if isinstance(data, dict):
                data = [data]
            
            width, height = image_size
            
            for item in data:
                if 'bbox_2d' in item:
                    bbox_rel = item['bbox_2d']
                    
                    # Qwen3-VL使用相对坐标0-1000，转换为绝对坐标
                    if max(bbox_rel) > 1000:
                        # 已经是绝对坐标
                        x1_abs = int(bbox_rel[0])
                        y1_abs = int(bbox_rel[1])
                        x2_abs = int(bbox_rel[2])
                        y2_abs = int(bbox_rel[3])
                    else:
                        # 相对坐标转绝对坐标
                        x1_abs = int(bbox_rel[0] / 1000 * width)
                        y1_abs = int(bbox_rel[1] / 1000 * height)
                        x2_abs = int(bbox_rel[2] / 1000 * width)
                        y2_abs = int(bbox_rel[3] / 1000 * height)
                    
                    # 确保坐标正确
                    if x1_abs > x2_abs:
                        x1_abs, x2_abs = x2_abs, x1_abs
                    if y1_abs > y2_abs:
                        y1_abs, y2_abs = y2_abs, y1_abs
                    
                    # 限制在图像范围内
                    x1_abs = max(0, min(x1_abs, width))
                    x2_abs = max(0, min(x2_abs, width))
                    y1_abs = max(0, min(y1_abs, height))
                    y2_abs = max(0, min(y2_abs, height))
                    
                    # 验证检测框大小
                    if x2_abs - x1_abs > 5 and y2_abs - y1_abs > 5:
                        detection = {
                            'bbox_2d_relative': bbox_rel,
                            'bbox_absolute': [x1_abs, y1_abs, x2_abs, y2_abs],
                            'label': item.get('label', 'carton'),
                            'confidence': item.get('confidence', 1.0)
                        }
                        detections.append(detection)
            
            # NMS去重
            detections = self.remove_duplicates(detections)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {output_text[:200]}")
        except Exception as e:
            logger.warning(f"解析失败: {e}")
        
        return detections
    
    def remove_duplicates(self, detections, iou_threshold=0.5):
        """使用NMS去除重复检测"""
        if len(detections) <= 1:
            return detections
        
        # 按面积排序
        detections_with_area = []
        for det in detections:
            bbox = det['bbox_absolute']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            detections_with_area.append((det, area))
        
        detections_with_area.sort(key=lambda x: x[1], reverse=True)
        
        suppressed = set()
        for i, (det_i, area_i) in enumerate(detections_with_area):
            if i in suppressed:
                continue
            
            for j, (det_j, area_j) in enumerate(detections_with_area[i+1:], i+1):
                if j in suppressed:
                    continue
                
                bbox_i = det_i['bbox_absolute']
                bbox_j = det_j['bbox_absolute']
                
                # 计算IoU
                x1_inter = max(bbox_i[0], bbox_j[0])
                y1_inter = max(bbox_i[1], bbox_j[1])
                x2_inter = min(bbox_i[2], bbox_j[2])
                y2_inter = min(bbox_i[3], bbox_j[3])
                
                if x1_inter < x2_inter and y1_inter < y2_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                    area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                    union_area = area_i + area_j - inter_area
                    
                    if union_area > 0:
                        iou = inter_area / union_area
                        if iou > iou_threshold:
                            suppressed.add(j)
        
        return [det for i, (det, _) in enumerate(detections_with_area) if i not in suppressed]
    
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
        
        # 保存YOLO格式
        if result['detections']:
            label_path = self.outputs_dir / 'labels' / f"{image_name}.txt"
            with open(label_path, 'w') as f:
                for det in result['detections']:
                    bbox_abs = det['bbox_absolute']
                    width = result['original_size']['width']
                    height = result['original_size']['height']
                    
                    x_center = (bbox_abs[0] + bbox_abs[2]) / 2 / width
                    y_center = (bbox_abs[1] + bbox_abs[3]) / 2 / height
                    box_width = (bbox_abs[2] - bbox_abs[0]) / width
                    box_height = (bbox_abs[3] - bbox_abs[1]) / height
                    
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        self.visualize_detections(result)
    
    def visualize_detections(self, result):
        try:
            image = Image.open(result['image_path']).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 
                     'purple', 'cyan', 'magenta', 'lime']
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            for i, det in enumerate(result['detections']):
                bbox = det['bbox_absolute']
                color = colors[i % len(colors)]
                label = det['label']
                confidence = det.get('confidence', 1.0)
                
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                             outline=color, width=3)
                
                text = f"{label} {confidence:.2f}"
                text_bbox = draw.textbbox((bbox[0], bbox[1] - 25), text, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, 
                              text_bbox[2]+2, text_bbox[3]+2], fill=color)
                draw.text((bbox[0], bbox[1] - 25), text, fill='white', font=font)
            
            image_name = Path(result['image_name']).stem
            vis_path = self.outputs_dir / 'images' / f"{image_name}_detection.jpg"
            image.save(vis_path, quality=95)
            
        except Exception as e:
            logger.error(f"可视化失败: {e}")
    
    def process_all_images(self):
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("没有找到图片")
            return
        
        logger.info(f"开始处理 {len(image_files)} 张图片...")
        
        results_summary = {
            'detection_type': '2D',
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'total_detections': 0,
            'results': []
        }
        
        for image_path in tqdm(image_files, desc="2D检测"):
            try:
                result = self.detect_cartons(image_path)
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
        logger.info(f"输出: {self.outputs_dir}")
        logger.info(f"{'='*50}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="纸箱2D检测系统")
    parser.add_argument('--model', '-m', default='./models', help='模型路径')
    parser.add_argument('--datasets', '-d', default='./datasets', help='数据集目录')
    parser.add_argument('--outputs', '-o', default='./outputs/2D', help='输出目录')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='设备')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("2D检测系统")
    logger.info(f"模型: {args.model}")
    logger.info(f"数据集: {args.datasets}")
    logger.info(f"输出: {args.outputs}")
    logger.info(f"设备: {args.device}")
    logger.info("="*60)
    
    detector = Carton2DDetectionSystem(
        model_path=args.model,
        datasets_dir=args.datasets,
        outputs_dir=args.outputs,
        device=args.device
    )
    
    detector.process_all_images()


if __name__ == "__main__":
    main()

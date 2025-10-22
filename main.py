# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B模型API服务
基于FastAPI框架实现的多模态大模型API接口
"""

import os
import io
import base64
import json
import time
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# 全局变量存储模型
model = None
processor = None
tokenizer = None

# 模型加载函数
def load_model(model_path: str = "./models"):
    """加载Qwen3-VL-4B模型"""
    global model, processor, tokenizer
    
    print("正在加载Qwen3-VL-4B模型...")
    start_time = time.time()
    
    try:
        # 尝试使用特定的Qwen3VL模型类
        try:
            from transformers import Qwen3VLForConditionalGeneration
            # 先加载配置并修改ROPE配置
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # 如果存在rope_scaling配置且有mrope_interleaved字段，尝试处理
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'rope_scaling'):
                if isinstance(config.text_config.rope_scaling, dict) and 'mrope_interleaved' in config.text_config.rope_scaling:
                    # 保留mrope_section但移除可能导致问题的mrope_interleaved字段
                    print("已处理ROPE配置中的mrope_interleaved字段")
            
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            print("使用Qwen3VLForConditionalGeneration加载模型成功")
        except Exception as e:
            print(f"使用Qwen3VLForConditionalGeneration加载失败: {str(e)}")
            # 如果失败，尝试使用Qwen2VLForConditionalGeneration
            try:
                from transformers import Qwen2VLForConditionalGeneration
                # 先加载配置并修改ROPE配置
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                
                # 如果存在rope_scaling配置且有mrope_interleaved字段，移除它
                if hasattr(config, 'text_config') and hasattr(config.text_config, 'rope_scaling'):
                    if isinstance(config.text_config.rope_scaling, dict) and 'mrope_interleaved' in config.text_config.rope_scaling:
                        # 移除可能导致问题的字段
                        config.text_config.rope_scaling.pop('mrope_interleaved', None)
                        print("已处理ROPE配置中的mrope_interleaved字段")
                
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    config=config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                print("使用Qwen2VLForConditionalGeneration加载模型成功")
            except Exception as e2:
                print(f"使用Qwen2VLForConditionalGeneration加载失败: {str(e2)}")
                # 如果还失败，尝试使用AutoModelForImageTextToText
                from transformers import AutoModelForImageTextToText
                # 先加载配置并修改ROPE配置
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                
                # 如果存在rope_scaling配置且有mrope_interleaved字段，移除它
                if hasattr(config, 'text_config') and hasattr(config.text_config, 'rope_scaling'):
                    if isinstance(config.text_config.rope_scaling, dict) and 'mrope_interleaved' in config.text_config.rope_scaling:
                        # 移除可能导致问题的字段
                        config.text_config.rope_scaling.pop('mrope_interleaved', None)
                        print("已处理ROPE配置中的mrope_interleaved字段")
                
                model = AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    config=config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                print("使用AutoModelForImageTextToText加载模型成功")
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        end_time = time.time()
        print(f"模型加载完成，耗时: {end_time - start_time:.2f}秒")
        return True
    
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

# 使用lifespan事件处理器替代已弃用的on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    model_dir = "./models"
    if os.path.exists(model_dir):
        load_model(model_dir)
    else:
        print(f"警告: 模型目录 {model_dir} 不存在")
        print("请确保已从魔搭社区下载Qwen3-VL-4B模型到该目录")
    
    yield
    
    # 关闭时清理资源
    global model, processor, tokenizer
    model = None
    processor = None
    tokenizer = None
    print("模型资源已释放")

# 创建FastAPI应用
app = FastAPI(
    title="Qwen3-VL-4B API",
    description="Qwen3-VL-4B多模态大模型API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class TextRequest(BaseModel):
    text: str
    history: Optional[List[Dict[str, str]]] = []
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8

class MultimodalRequest(BaseModel):
    text: str
    image: Optional[str] = None  # base64编码的图片
    history: Optional[List[Dict[str, str]]] = []
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8

# 根路径
@app.get("/")
async def root():
    return {"message": "Qwen3-VL-4B API服务已启动"}

# 健康检查
@app.get("/health")
async def health_check():
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy", "model": "Qwen3-VL-4B"}

# 文本生成接口
@app.post("/api/v1/chat/text")
async def text_generation(request: TextRequest):
    """纯文本对话接口"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 构建对话历史
        messages = []
        if request.history:
            messages.extend(request.history)
        
        messages.append({"role": "user", "content": request.text})
        
        # 使用tokenizer处理文本
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成响应
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "response": response,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本生成失败: {str(e)}")

# 多模态对话接口
@app.post("/api/v1/chat/multimodal")
async def multimodal_generation(request: MultimodalRequest):
    """多模态对话接口（文本+图片）"""
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 构建消息内容
        content = [{"type": "text", "text": request.text}]
        
        # 如果有图片，添加到消息中
        if request.image:
            # 解码base64图片
            image_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_data))
            content.append({"type": "image", "image": image})
        
        # 构建对话历史
        messages = []
        if request.history:
            messages.extend(request.history)
        
        messages.append({"role": "user", "content": content})
        
        # 使用processor处理输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 生成响应
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "response": response,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多模态生成失败: {str(e)}")

# 图片上传和多模态对话接口
@app.post("/api/v1/chat/upload")
async def upload_and_chat(
    file: UploadFile = File(...),
    text: str = Form(...),
    max_length: int = Form(2048),
    temperature: float = Form(0.7),
    top_p: float = Form(0.8)
):
    """上传图片并进行多模态对话"""
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 检查文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="上传的文件不是图片")
        
        # 读取图片数据
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 构建消息内容
        content = [
            {"type": "text", "text": text},
            {"type": "image", "image": image}
        ]
        
        # 构建消息
        messages = [{"role": "user", "content": content}]
        
        # 使用processor处理输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 生成响应
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "response": response,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 模型信息接口
@app.get("/api/v1/model/info")
async def model_info():
    """获取模型信息"""
    global model, tokenizer
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_name": "Qwen3-VL-4B",
        "model_type": "multimodal",
        "parameters": "4B",
        "device": str(model.device),
        "status": "loaded"
    }

# 启动服务
if __name__ == "__main__":
    # 启动API服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
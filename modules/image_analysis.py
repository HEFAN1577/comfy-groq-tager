import logging
import base64
from io import BytesIO
from PIL import Image
import httpx
import json
import os
from typing import Union

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def encode_image_to_base64(image: Union[Image.Image, str]) -> str:
    """将图片转换为base64编码"""
    if isinstance(image, str):
        with open(image, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    else:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_prompt(image: Union[Image.Image, str]) -> str:
    """
    使用Llama-3.2-90b-vision-preview模型分析图片并生成提示词
    Args:
        image: PIL.Image对象或图片文件路径
    Returns:
        str: 生成的提示词描述
    """
    try:
        # 将图片转换为base64
        base64_image = encode_image_to_base64(image)
        
        # 构建请求消息 - 移除 system message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "你是一个专业的图片分析助手。请详细分析这张图片，生成详细的提示词描述，包括：1. 主要内容和主体 2. 艺术风格 3. 构图方式 4. 光影效果 5. 色彩搭配 6. 其他特殊细节。请用中文回答，尽可能详细和专业。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        # 发送请求到Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"分析图片时出错: {error_msg}"

    except Exception as e:
        error_msg = f"生成图片提示词时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg
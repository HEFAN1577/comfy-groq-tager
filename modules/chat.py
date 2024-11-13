from groq import Groq
import logging
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# 从环境变量获取API密钥
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("未找到API密钥，请在.env文件中设置GROQ_API_KEY")

client = Groq(api_key=api_key)

def encode_image_to_base64(image):
    """将图片转换为base64格式，并进行压缩"""
    if image is None:
        return None
        
    try:
        # 转换为RGB模式（如果是RGBA）
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # 设置最大尺寸为 1200 像素
        max_size = 1200
        ratio = min(max_size/float(image.size[0]), max_size/float(image.size[1]))
        if ratio < 1:
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 保持高质量设置
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
        
    except Exception as e:
        logger.error(f"图片编码错误: {e}")
        return None

def chat_with_ai(message: str, history=None, image=None) -> str:
    """
    与Groq AI进行对话
    """
    try:
        # 准备消息历史
        messages = []
        
        if history:
            for human, assistant in history:
                messages.append({"role": "user", "content": human})
                messages.append({"role": "assistant", "content": assistant})
        
        # 处理图片
        if image is not None:
            base64_image = encode_image_to_base64(image)
            if base64_image:
                # 构建包含图片的消息
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"请用中文详细描述这张图片。{message if message else ''}"
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"{message} (图片处理失败)"
                })
        else:
            # 纯文本消息
            messages.append({
                "role": "user",
                "content": message + " (请用中文回复)"
            })
        
        logger.info("正在发送请求到Groq API...")
        
        # 调用Groq API，使用新的模型名称
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",  # 更新模型名称
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        response = completion.choices[0].message.content
        
        # 如果响应不是中文，请求中文翻译
        if not any('\u4e00' <= char <= '\u9fff' for char in response):
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "请用中文重新表达上述内容"
            })
            
            completion = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",  # 这里也更新模型名称
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )
            response = completion.choices[0].message.content
        
        return response
        
    except Exception as e:
        logger.error(f"API调用错误: {e}")
        logger.error(f"错误详情: {str(e)}")
        return f"抱歉，发生了错误: {str(e)}"
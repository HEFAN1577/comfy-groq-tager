import os
import logging
import tempfile
import httpx
from PIL import Image
import cv2
import numpy as np
import io
import base64
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from modules.audio_analysis import extract_audio_from_video, transcribe_audio

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class RateLimiter:
    def __init__(self, rpm_limit):
        self.min_interval = 60.0 / rpm_limit
        self.last_request_time = 0

    def wait_if_needed(self):
        """等待必要的时间以遵守速率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

rate_limiter = RateLimiter(30)  # 30 RPM限制，即每2秒一个请求

def extract_frames(video_path: str) -> list:
    """提取视频关键帧"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 每隔一定间隔提取帧
        interval = max(1, total_frames // 10)  # 最多提取10帧
        
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # 调整图片大小
                height, width = frame.shape[:2]
                if width > 1920:
                    scale = 1920 / width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)
                
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"提取视频帧时出错: {e}")
        raise

def frame_to_base64(frame: np.ndarray) -> str:
    """将视频帧转换为base64编码"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logger.error(f"转换帧到base64时出错: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=2, max=5),  # 减少等待时间
    reraise=True
)
def analyze_frame(frame: np.ndarray) -> str:
    """分析单个视频帧"""
    try:
        rate_limiter.wait_if_needed()
        base64_image = frame_to_base64(frame)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请用简短的中文描述这个画面中的主要内容和场景，不要太详细，只需要概括性描述。"
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
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 150,
            "stream": False
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API请求失败: {response.status_code}")
    except Exception as e:
        logger.error(f"分析帧时出错: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=2, max=5),  # 减少等待时间
    reraise=True
)
def generate_final_summary(frame_descriptions: list, audio_content: str) -> str:
    """生成最终的视频总结，整合视觉和音频信息"""
    try:
        rate_limiter.wait_if_needed()
        
        prompt = f"""请基于以下视频的视觉和音频信息，生成一个全面的内容总结：

1. 视频画面描述：
{' '.join(frame_descriptions)}

2. 视频音频内容：
{audio_content}

请生成一个结构化的总结，包含：
1. 视频的主要内容和主题
2. 视频的场景和环境描述
3. 视频中的对话或声音内容要点
4. 视频的风格和表现形式
5. 视频可能的目的或用途

请用中文输出，使用简洁清晰的语言，确保总结既包含视觉信息，也包含音频信息。如果视觉和音频信息存在关联，请特别指出。"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.2-90b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API请求失败: {response.status_code}")
    except Exception as e:
        logger.error(f"生成最终总结时出错: {e}")
        raise

def process_video_summary(video_file) -> str:
    """处理视频并生成综合总结"""
    if video_file is None:
        return "请上传视频文件"
        
    try:
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video_path = temp_video.name
            with open(video_file.name, 'rb') as f:
                temp_video.write(f.read())
        
        logger.info(f"开始处理视频总结: {video_file.name}")
        
        # 1. 提取并分析视频帧
        frames = extract_frames(temp_video_path)
        frame_descriptions = []
        for i, frame in enumerate(frames):
            try:
                description = analyze_frame(frame)
                if description:
                    frame_descriptions.append(description)
                logger.info(f"已完成第 {i+1}/{len(frames)} 帧的分析")
            except Exception as e:
                logger.error(f"处理第 {i} 帧时出错: {e}")
                continue
        
        # 2. 提取并分析音频内容
        try:
            audio_path = extract_audio_from_video(temp_video_path)
            audio_content = transcribe_audio(audio_path)
            logger.info("已完成音频内容分析")
        except Exception as e:
            logger.error(f"处理音频时出错: {e}")
            audio_content = "无法提取音频内容"
        
        # 3. 生成综合总结
        if frame_descriptions or audio_content != "无法提取音频内容":
            summary = generate_final_summary(frame_descriptions, audio_content)
        else:
            summary = "无法从视频中提取有效信息"
        
        # 清理临时文件
        os.unlink(temp_video_path)
        
        return summary
        
    except Exception as e:
        logger.error(f"生成视频总结时出错: {e}")
        return f"生成视频总结时发生错误: {str(e)}"
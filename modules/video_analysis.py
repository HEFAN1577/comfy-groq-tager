import cv2
import numpy as np
import logging
import os
from typing import List, Tuple
import tempfile
import base64
from io import BytesIO
import httpx
from PIL import Image
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
RPM_LIMIT = 15  # Groq API 每分钟请求限制
MIN_REQUEST_INTERVAL = 60.0 / RPM_LIMIT  # 每次请求的最小间隔时间（秒）

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
            logger.info(f"等待 {sleep_time:.2f} 秒以遵守速率限制...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

# 创建速率限制器实例
rate_limiter = RateLimiter(RPM_LIMIT)

def extract_frames(video_path: str, interval: int = 15) -> List[np.ndarray]:
    """
    从视频中提取帧
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 每秒提取1帧
        interval = max(1, int(fps))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # 调整图片大小，提高处理速度
                height, width = frame.shape[:2]
                if width > 1920:
                    scale = 1920 / width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        return frames
    except Exception as e:
        logger.error(f"提取视频帧时出错: {e}")
        raise

def frame_to_base64(frame: np.ndarray) -> str:
    """
    将视频帧转换为base64编码
    """
    try:
        # 将OpenCV的BGR格式转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # 转换为base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"转换帧到base64时出错: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def extract_text_from_frame(frame: np.ndarray) -> str:
    """
    使用Llama模型从帧中提取文字，带有重试机制
    """
    try:
        rate_limiter.wait_if_needed()
        base64_image = frame_to_base64(frame)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """请严格按照以下要求识别图片中的字幕文字：
1. 只识别图片中实际存在的字幕文本，不要进行推测或补充
2. 如果图片中没有字幕，必须返回空字符串
3. 如果字幕不完整或无法确定，返回空字符串
4. 不要包含任何水印、logo或其他非字幕文字
5. 不要添加任何标点符号
6. 不要对内容进行修改或润色
7. 只返回字幕文本，不要包含任何解释或描述
8. 如果字幕分多行，按原样返回，不要主动合并

注意：宁可不返回，也不要返回不确定的内容。
只有在100%确定看到字幕内容时才返回。

示例正确输出：
"这是测试字幕"
或
"" (当没有看到明确的字幕时)
"""
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
            "temperature": 0.1,  # 降低温度，使输出更加保守
            "max_tokens": 100,
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
                text = result['choices'][0]['message']['content'].strip()
                
                # 增强的文本清理逻辑
                # 如果文本包含这些词，可能是模型的解释而不是实际字幕
                ignore_phrases = [
                    "图片中", "显示", "字幕是", "内容是", "文字是",
                    "我看到", "这是", "这个", "有", "没有",
                    "字幕内容", "文本", "识别到"
                ]
                
                for phrase in ignore_phrases:
                    if phrase in text:
                        return ""
                
                # 如果文本太长，可能是模型的解释
                if len(text) > 50:
                    return ""
                
                # 去除引号
                text = text.strip('"').strip("'")
                
                return text.strip() if text and len(text) > 1 else ""
            else:
                raise Exception(f"API请求失败: {response.status_code}")

    except Exception as e:
        logger.error(f"从帧提取文本时出错: {e}")
        raise

def extract_subtitles(video_path: str, interval_seconds: float = 1.0) -> List[Tuple[float, str]]:
    """
    从视频中提取字幕，返回带时间戳的字幕列表
    Args:
        video_path: 视频文件路径
        interval_seconds: 抽帧间隔（秒）
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 根据指定的秒数计算帧间隔
        interval = int(fps * interval_seconds)
        
        subtitles = []
        frame_count = 0
        processed_count = 0
        
        logger.info(f"开始处理视频，总帧数: {total_frames}, FPS: {fps}, 提取间隔: {interval}帧 ({interval_seconds}秒/帧)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                timestamp = frame_count / fps
                
                # 调整图片大小
                height, width = frame.shape[:2]
                if width > 1920:
                    scale = 1920 / width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                try:
                    # 提取文本
                    text = extract_text_from_frame(frame)
                    if text:  # 只添加有文本的帧
                        subtitles.append((timestamp, text))
                    
                    processed_count += 1
                    logger.info(f"处理进度: {processed_count}/{total_frames//interval} (时间: {timestamp:.1f}s)")
                except Exception as e:
                    logger.error(f"处理帧 {frame_count} 时出错: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        return subtitles
        
    except Exception as e:
        logger.error(f"提取字幕时出错: {e}")
        raise

def process_video(video_file, interval_seconds: float = 1.0) -> str:
    """
    处理上传的视频文件
    """
    if video_file is None:
        return "请上传视频文件"
        
    try:
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
            with open(video_file.name, 'rb') as f:
                temp_file.write(f.read())
        
        logger.info(f"开始处理视频: {video_file.name}, 抽帧间隔: {interval_seconds}秒")
        
        # 初始化去重集合
        seen_texts = set()
        
        # 提取字幕
        subtitles = extract_subtitles(temp_path, interval_seconds)
        
        # 删除临时文件
        os.unlink(temp_path)
        
        # 格式化输出
        if subtitles:
            result = []
            
            for timestamp, text in subtitles:
                # 清理文本
                text = text.strip()
                text = text.replace("抖音", "").strip()
                
                # 跳过空文本
                if not text:
                    continue
                    
                # 将文本按句号、问号等分割成单独的句子
                sentences = [s.strip() for s in text.replace('。', '。\n').replace('？', '？\n').replace('！', '！\n').split('\n')]
                sentences = [s for s in sentences if s]  # 移除空字符串
                
                for sentence in sentences:
                    # 跳过太短的句子
                    if len(sentence) < 2:
                        continue
                        
                    # 跳过已经出现过的句子
                    if sentence in seen_texts:
                        continue
                        
                    # 检查是否与已有句子过于相似
                    skip = False
                    for seen_text in seen_texts:
                        if _similar_text(sentence, seen_text):
                            skip = True
                            break
                    if skip:
                        continue
                    
                    seen_texts.add(sentence)
                    result.append(f"[{int(timestamp)}s] {sentence}")
            
            return "\n".join(result) if result else "未能从视频中提取到有效字幕"
        else:
            return "未能从视频中提取到字幕"
            
    except Exception as e:
        logger.error(f"处理视频时出错: {e}")
        return f"处理视频时发生错误: {str(e)}"
    finally:
        # 确保在处理结束后清理缓存
        if 'seen_texts' in locals():
            seen_texts.clear()

def _similar_text(text1: str, text2: str) -> bool:
    """
    检查两段文本是否相似
    使用简单的似度计算：相同字符的比例
    """
    if abs(len(text1) - len(text2)) > 5:
        return False
        
    # 计算相同字符的数量
    same_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
    max_len = max(len(text1), len(text2))
    
    # 如果相同字符比例超过80%，认为是相似文本
    return same_chars / max_len > 0.8
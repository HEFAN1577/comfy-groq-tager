import os
import logging
import tempfile
import subprocess
from groq import Groq
import json

logger = logging.getLogger(__name__)

# 初始化 Groq 客户端
client = Groq()

# 获取 ffmpeg 路径
FFMPEG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ffmpeg", "bin", "ffmpeg.exe")

def extract_audio_from_video(video_path: str) -> str:
    """
    从视频中提取音频
    """
    try:
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_audio:
            temp_audio_path = temp_audio.name
            
        # 使用项目目录下的 ffmpeg 提取音频
        cmd = [
            FFMPEG_PATH,
            '-i', video_path,
            '-vn',  # 不处理视频
            '-acodec', 'copy',  # 复制音频流
            temp_audio_path,
            '-y'
        ]
        
        # 执行命令
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg 执行失败: {process.stderr.decode()}")
            
        return temp_audio_path
    except Exception as e:
        logger.error(f"提取音频时出错: {e}")
        raise

def transcribe_audio(audio_path: str) -> str:
    """
    使用 whisper-large-v3 模型转录音频
    """
    try:
        with open(audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
            
            # 解析返回的 JSON 数据
            result = []
            # 将返回结果转换为字典
            response_dict = transcription if isinstance(transcription, dict) else transcription.model_dump()
            
            # 从字典中获取 segments
            segments = response_dict.get('segments', [])
            
            for segment in segments:
                # 从字典中获取开始时间和文本
                start_time = int(float(segment.get('start', 0)))  # 获取开始时间（秒）
                text = segment.get('text', '').strip()  # 获取文本内容
                if text:  # 如果文本不为空
                    result.append(f"[{start_time}s] {text}")
            
            return "\n".join(result) if result else "未能识别到任何语音内容"
                
    except Exception as e:
        logger.error(f"转录音频时出错: {e}")
        return f"转录音频时出错: {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(audio_path):
            os.unlink(audio_path)

def process_video_audio(video_file) -> str:
    """
    处理视频的音频内容
    """
    if video_file is None:
        return "请上传视频文件"
        
    try:
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video_path = temp_video.name
            # 保存上传的视频到临时文件
            with open(video_file.name, 'rb') as f:
                temp_video.write(f.read())
        
        logger.info(f"开始处理视频音频: {video_file.name}")
        
        # 提取音频
        audio_path = extract_audio_from_video(temp_video_path)
        
        # 转录音频
        transcription = transcribe_audio(audio_path)
        
        # 清理临时视频文件
        os.unlink(temp_video_path)
        
        return transcription
        
    except Exception as e:
        logger.error(f"处理视频音频时出错: {e}")
        return f"处理视频音频时发生错误: {str(e)}"
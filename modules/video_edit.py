import os
import logging
import tempfile
import subprocess

logger = logging.getLogger(__name__)

def process_video_edit(video_file, start_time: float, end_time: float) -> str:
    """
    剪辑视频
    Args:
        video_file: 上传的视频文件
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
    Returns:
        剪辑后的视频文件路径
    """
    if video_file is None:
        return None
    
    try:
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            input_path = temp_input.name
            with open(video_file.name, 'rb') as f:
                temp_input.write(f.read())
        
        # 创建输出文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            output_path = temp_output.name
        
        logger.info(f"开始剪辑视频: {video_file.name}, 时间范围: {start_time}s - {end_time}s")
        
        # 构建 ffmpeg 命令
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-i', input_path,  # 输入文件
            '-ss', str(start_time),  # 开始时间
            '-t', str(duration),  # 持续时间
            '-c:v', 'libx264',  # 视频编码器
            '-c:a', 'aac',  # 音频编码器
            '-strict', 'experimental',
            '-b:a', '192k',  # 音频比特率
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        # 执行命令
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg 执行失败: {process.stderr.decode()}")
        
        # 删除输入临时文件
        os.unlink(input_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"剪辑视频时出错: {e}")
        if 'input_path' in locals() and os.path.exists(input_path):
            os.unlink(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)
        return None 
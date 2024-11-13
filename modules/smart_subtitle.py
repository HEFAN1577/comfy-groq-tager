import os
import logging
import tempfile
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
from modules.video_analysis import extract_subtitles
from modules.audio_analysis import extract_audio_from_video, transcribe_audio

logger = logging.getLogger(__name__)

def similar(a: str, b: str) -> float:
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def merge_subtitles(video_subs: List[Tuple[float, str]], audio_subs: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    合并并优化视频和音频字幕
    """
    merged = []
    used_audio = set()
    seen_texts = set()  # 为每次处理创建新的去重集合
    
    # 将音频字幕转换为字典，便于查找
    audio_dict = {int(timestamp): text for timestamp, text in audio_subs}
    
    for v_time, v_text in video_subs:
        v_time_int = int(v_time)
        best_match = None
        best_score = 0
        matched_time = None
        
        # 在音频字幕中寻找最佳匹配
        for a_time in range(max(0, v_time_int - 2), v_time_int + 3):  # 检查前后2秒
            if a_time in audio_dict and a_time not in used_audio:
                a_text = audio_dict[a_time]
                score = similar(v_text, a_text)
                if score > best_score and score > 0.3:  # 设置最小相似度阈值
                    best_score = score
                    best_match = a_text
                    matched_time = a_time
        
        if best_match:
            # 如果找到匹配的音频字幕，选择较长的那个
            used_audio.add(matched_time)
            final_text = best_match if len(best_match) > len(v_text) else v_text
            if final_text not in seen_texts:  # 检查是否已存在
                seen_texts.add(final_text)
                merged.append((v_time, final_text))
        else:
            # 如果没有匹配的音频字幕，保留视频字幕
            if v_text not in seen_texts:  # 检查是否已存在
                seen_texts.add(v_text)
                merged.append((v_time, v_text))
    
    # 添加未匹配的音频字幕
    for a_time, a_text in audio_subs:
        a_time_int = int(a_time)
        if a_time_int not in used_audio and a_text not in seen_texts:
            # 检查是否与现有字幕重叠
            overlap = False
            for m_time, _ in merged:
                if abs(m_time - a_time) < 2:  # 2秒内的重叠
                    overlap = True
                    break
            if not overlap:
                seen_texts.add(a_text)
                merged.append((a_time, a_text))
    
    # 按时间排序
    merged.sort(key=lambda x: x[0])
    
    # 清理并去重
    final_subs = []
    seen_texts.clear()  # 重新清空集合
    
    for time, text in merged:
        # 清理文本
        text = text.strip()
        text = text.replace("抖音", "").strip()
        
        # 跳过空文本或太短的文本
        if not text or len(text) < 2:
            continue
        
        # 检查是否与已有文本过于相似
        is_similar = False
        for seen_text in seen_texts:
            if similar(text, seen_text) > 0.7:  # 设置相似度阈值
                is_similar = True
                break
        
        if not is_similar:
            seen_texts.add(text)
            final_subs.append((time, text))
    
    return final_subs

def process_smart_subtitle(video_file) -> str:
    """
    智能处理视频字幕，结合视频帧OCR和音频识别结果
    """
    if video_file is None:
        return "请上传视频文件"
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video_path = temp_video.name
            with open(video_file.name, 'rb') as f:
                temp_video.write(f.read())
        
        logger.info(f"开始智能提取字幕: {video_file.name}")
        
        # 1. 提取视频字幕
        video_subtitles = extract_subtitles(temp_video_path, interval_seconds=1.0)
        logger.info("完成视频字幕提取")
        
        # 2. 提取音频字幕
        audio_path = extract_audio_from_video(temp_video_path)
        audio_content = transcribe_audio(audio_path)
        
        # 解析音频内容为时间戳格式
        audio_subtitles = []
        for line in audio_content.split('\n'):
            if line.startswith('[') and ']' in line:
                try:
                    time_str = line[1:line.index('s]')]
                    text = line[line.index(']')+1:].strip()
                    audio_subtitles.append((float(time_str), text))
                except:
                    continue
        
        logger.info("完成音频字幕提取")
        
        # 3. 合并并优化字幕
        merged_subtitles = merge_subtitles(video_subtitles, audio_subtitles)
        
        # 4. 格式化输出
        if merged_subtitles:
            result = []
            for timestamp, text in merged_subtitles:
                result.append(f"[{int(timestamp)}s] {text}")
            return "\n".join(result)
        else:
            return "未能提取到有效字幕"
        
    except Exception as e:
        logger.error(f"智能提取字幕时出错: {e}")
        return f"处理出错: {str(e)}"
    finally:
        # 清理临时文件
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path) 
import os
import logging
import tempfile
import subprocess
import json
from typing import List, Dict, Tuple
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, AudioFileClip, CompositeVideoClip
from modules.video_summary import process_video_summary
from modules.video_analysis import extract_frames, extract_text_from_frame
from groq import Groq
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# 预定义的背景音乐选项
MUSIC_OPTIONS = {
    "温馨": "assets/music/warm.mp3",
    "活力": "assets/music/energetic.mp3",
    "搞笑": "assets/music/funny.mp3",
    "严肃": "assets/music/serious.mp3",
    "文艺": "assets/music/artistic.mp3"
}

class VideoSegment:
    def __init__(self, path: str, start: float, duration: float, score: float):
        self.path = path
        self.start = start
        self.duration = duration
        self.score = score
        self.description = ""
        self.visual_score = 0.0
        self.audio_score = 0.0
        self.transition_score = 0.0

async def analyze_video_content(video_path: str) -> List[Dict]:
    """异步分析视频内容"""
    scenes = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # 每秒末尾帧
            for second in range(int(duration)):
                frame_pos = min((second + 1) * fps - 1, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if ret:
                    # 调整图片大小
                    height, width = frame.shape[:2]
                    if width > 1920:
                        scale = 1920 / width
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    
                    # 提交任务到线程池
                    future = executor.submit(extract_text_from_frame, frame)
                    futures.append((second, future))
            
            # 收集结果
            for second, future in futures:
                try:
                    description = future.result()
                    if description:
                        scenes.append({
                            "time": second,
                            "description": description
                        })
                    logger.info(f"已分析 {second}/{int(duration)} 秒")
                except Exception as e:
                    logger.error(f"处理第 {second} 秒时出错: {e}")
        
        cap.release()
        return scenes
    except Exception as e:
        logger.error(f"分析视频内容时出错: {e}")
        raise

async def match_scenes_with_prompt(scenes: List[Dict], prompt: str) -> List[Tuple[float, float, float]]:
    """异步评估场景匹配度"""
    try:
        scores = []
        for scene in scenes:
            # 评估视觉相关性
            visual_response = await asyncio.to_thread(
                client.chat.completions.create,
                model="llama-3.2-90b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""评估场景与目标的视觉相关性（0-1分）：
场景：{scene['description']}
目标：{prompt}
只返回分数"""
                    }
                ],
                temperature=0.1
            )
            
            # 评估情感匹配度
            emotion_response = await asyncio.to_thread(
                client.chat.completions.create,
                model="llama-3.2-90b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""评估场景与目标的情感匹配度（0-1分）：
场景：{scene['description']}
目标：{prompt}
只返回分数"""
                    }
                ],
                temperature=0.1
            )
            
            # 评估叙事连贯性
            narrative_response = await asyncio.to_thread(
                client.chat.completions.create,
                model="llama-3.2-90b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""评估场景与整体叙事的连贯性（0-1分）：
场景：{scene['description']}
目标：{prompt}
只返回分数"""
                    }
                ],
                temperature=0.1
            )
            
            try:
                visual_score = float(visual_response.choices[0].message.content.strip())
                emotion_score = float(emotion_response.choices[0].message.content.strip())
                narrative_score = float(narrative_response.choices[0].message.content.strip())
                scores.append((visual_score, emotion_score, narrative_score))
            except:
                scores.append((0.0, 0.0, 0.0))
        
        return scores
    except Exception as e:
        logger.error(f"场景匹配评分时出错: {e}")
        raise

def polish_prompt(prompt: str) -> str:
    """润色用户输入的提示文本"""
    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b",
            messages=[
                {
                    "role": "user",
                    "content": f"""请对以下视频内容描述进行专业的润色和完善：

原始描述：{prompt}

要求：
1. 使用专业的视频制作术语
2. 添加具体的镜头要求（远景、特写等）
3. 指定画面风格和氛围
4. 说明转场和节奏要求
5. 描述音乐和音效需求
6. 保持中文表达

请直接返回润色后的文本。"""
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"润色文本时出错: {e}")
        return prompt

async def process_video_compose(
    video_files,
    prompt: str,
    style: str,
    target_duration: float,
    transition_type: str,
    progress_callback=None
) -> str:
    """异步处理视频合成"""
    if not video_files:
        return "请上传视频文件", None
    
    try:
        # 润色提示文本
        if progress_callback:
            progress_callback("正在优化内容描述...")
        polished_prompt = polish_prompt(prompt)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 并行分析所有视频
            all_segments = []
            analysis_tasks = []
            
            for i, video_file in enumerate(video_files):
                if progress_callback:
                    progress_callback(f"正在分析第 {i+1}/{len(video_files)} 个视频...")
                
                # 保存视频到临时文件
                temp_path = os.path.join(temp_dir, f"video_{i}.mp4")
                with open(video_file.name, 'rb') as f:
                    video_data = f.read()
                with open(temp_path, 'wb') as f:
                    f.write(video_data)
                
                # 创建异步分析任务
                task = analyze_video_content(temp_path)
                analysis_tasks.append(task)
            
            # 等待所有分析完成
            scenes_list = await asyncio.gather(*analysis_tasks)
            
            # 评分和创建片段
            for i, scenes in enumerate(scenes_list):
                scores = await match_scenes_with_prompt(scenes, polished_prompt)
                for scene, (visual, emotion, narrative) in zip(scenes, scores):
                    # 计算综合得分
                    total_score = (visual * 0.4 + emotion * 0.3 + narrative * 0.3)
                    if total_score > 0.6:
                        segment = VideoSegment(
                            path=os.path.join(temp_dir, f"video_{i}.mp4"),
                            start=scene['time'],
                            duration=1.0,
                            score=total_score
                        )
                        segment.description = scene['description']
                        segment.visual_score = visual
                        segment.audio_score = emotion
                        segment.transition_score = narrative
                        all_segments.append(segment)
            
            if not all_segments:
                return "未找到符合要求的视频片段", None
            
            # 智能排序和选择片段
            all_segments.sort(key=lambda x: (x.score, x.transition_score), reverse=True)
            selected_segments = []
            current_duration = 0
            
            for segment in all_segments:
                if current_duration + segment.duration <= target_duration:
                    selected_segments.append(segment)
                    current_duration += segment.duration
                if current_duration >= target_duration:
                    break
            
            if progress_callback:
                progress_callback("正在合成视频...")
            
            # 合成视频
            clips = []
            for i, segment in enumerate(selected_segments):
                clip = VideoFileClip(segment.path).subclip(
                    segment.start,
                    segment.start + segment.duration
                )
                
                # 应用风格效果
                if style == "温馨":
                    clip = clip.fx(vfx.colorx, 1.1)
                elif style == "活力":
                    clip = clip.fx(vfx.colorx, 1.2)
                elif style == "文艺":
                    clip = clip.fx(vfx.blackwhite)
                
                if i > 0 and transition_type != "无":
                    clips[-1], clip = apply_transition(clips[-1], clip, transition_type)
                
                clips.append(clip)
            
            # 合并视频片段
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # 添加背景音乐
            if style in MUSIC_OPTIONS and os.path.exists(MUSIC_OPTIONS[style]):
                background_music = AudioFileClip(MUSIC_OPTIONS[style])
                # 循环音乐以匹配视频长度
                if background_music.duration < final_clip.duration:
                    n_loops = int(np.ceil(final_clip.duration / background_music.duration))
                    background_music = concatenate_videoclips([background_music] * n_loops)
                # 裁剪音乐以匹配视频长度
                background_music = background_music.subclip(0, final_clip.duration)
                # 调整音量
                background_music = background_music.volumex(0.3)
                # 合并音频
                final_clip = final_clip.set_audio(
                    CompositeVideoClip([final_clip, background_music]).audio
                )
            
            # 导出结果
            output_path = os.path.join(temp_dir, "output.mp4")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac'
            )
            
            return "视频合成完成！", output_path
            
    except Exception as e:
        logger.error(f"合成视频时出错: {e}")
        return f"处理出错: {str(e)}", None
import gradio as gr
from modules.chat import chat_with_ai
import logging
from PIL import Image
import io
from modules.image_analysis import get_image_prompt
import os
from modules.video_analysis import process_video
from enum import Enum
from modules.audio_analysis import process_video_audio
from modules.video_summary import process_video_summary

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessMode(Enum):
    FAST = 1.3    # 快速模式：1.3秒/帧
    STANDARD = 1.0  # 标准模式：1秒/帧
    PRECISE = 0.7   # 精准模式：0.7秒/帧

def chat_with_context(message, history, image=None):
    """处理聊天和图片"""
    try:
        # 如果有图片，先显示在界面上
        if image is not None:
            logger.info("处理上传的图片...")
        
        # 调用API
        bot_response = chat_with_ai(message, history, image)
        history.append((message, bot_response))
        return "", history, None  # 清空输入框和图片
    except Exception as e:
        logger.error(f"处理消息出错: {e}")
        return "", history + [(message, f"抱歉，发生了错误: {str(e)}")], None

def analyze_images(files):
    """批量分析图片并生成提示词"""
    try:
        results = []
        if not files:  # 检查是否有上传文件
            return "请先上传图片文件"
            
        for file in files:
            if file is not None:  # 检查文件是否存在
                logger.info(f"正在处理文件: {file.name}")
                prompt = get_image_prompt(file.name)  # 直接传入文件路径
                results.append(f"文件: {os.path.basename(file.name)}\n提示词描述:\n{prompt}\n{'='*50}\n")
        
        return "\n".join(results) if results else "没有找到有效的图片文件"
    except Exception as e:
        logger.error(f"批量处理图片时出错: {e}")
        return f"处理图片时发生错误: {str(e)}"

def process_video_with_mode(video_file, mode: str) -> str:
    """
    根据选择的模式处理视频
    """
    if video_file is None:
        return "请上传视频文件"
    
    # 根据选择的模式设置间隔
    mode_map = {
        "快速": VideoProcessMode.FAST.value,
        "标准": VideoProcessMode.STANDARD.value,
        "精准": VideoProcessMode.PRECISE.value
    }
    interval = mode_map.get(mode, VideoProcessMode.STANDARD.value)
    
    return process_video(video_file, interval)

# 在创建界面之前添加CSS样式
css = """
    .logo-container img {
        transition: transform 0.3s ease;
    }
    .logo-container img:hover {
        transform: scale(1.1);
    }
"""

# 在文件顶部添加正确的 base64 图片数据
BILIBILI_LOGO = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAYAAABWzo5XAAAABHNCSVQICAgIfAhkiAAAAhFJREFUOI2llD9oE1Ecxz/v3eUSW0tJLi2hxT9QRxWJqC1CKSjFQYqgg0MdxEEQHEpBpEPp4uBQcHBwqh0c7GKhTirWsYPiYBdB0EGwgmJrKJe7S3J/HO5yJr3cpQZ/0+O93/f7ed/fe+/BktIAyP6c5vHLV+i6TrVaJZlMksvl2L0rQ3v7GhzHwbZtTNMkn89TKpVQVZVIJEJPTw+qqrYA1RQBRkZGGBwcxDRNisViEyQejxOLxVhaWmJ+fp5cLgfA+Pg4vu+TTqcxDIPBwUFUVW0tFgqFcF0XwzBQFAVFUTAMA9d1CYVCAJTLZVKpFADJZBLf98nn8/T19TE1NUUkEmkt8jwPgGg0SigUIhwOE4/H8TwPgEQiQbFYxPM8TNMkHo+Ty+WYnZ3l3r17rUHNUVUVy7LQNI1UKoVlWaiqCsDMzAzxeJxsNsvx48cZGxtjenqaaDTaGlQoFNA0DU3TKBQKAOi6jqZp6LoOwPnz5zl48CCO4zA8PNwANYJ0XScSiVCtVhkaGmJ5eZlKpYLrugCk02nK5TLVapXLly5y5/atFaAGUKVS4cyZs+zdd4Dp6WkOHTrEqVOnuHr1Kq7rcvPmTc6dO4fruty/P8nExAQAjuPg+z7A/0GWZfHw4SMePHhIpVJB0zR27NjB2NgYJ06cYO3atTiOg23bVCoVstks27Zta+jxL/UNzYiVpXP6jvgAAAAASUVORK5CYII="

MAOMI_LOGO = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAASCAYAAABWzo5XAAAABHNCSVQICAgIfAhkiAAAAZtJREFUOI2t1D9rFFEUBfDfzM7uJhvdxSRGkYgasLCxsLGxsbCw9DP4AfwKFhYWQtBCEEQQGwsRJJWVjYVYaLAJKgasYmGEoEaJ2d15z8Zdd2d3Z0bwwOvmvnvPu/fccx8jKaUwxFVcwQzO4zCWsYBX+Jh9Q2UkpTSBh7iN8YLvO+7jSUrpT6UxpTSJt7hW4/yMeXzFJk7gIs7iGW6mlH43jRN4h/M5voM5PEgp/e0SHsNDXM0mX+BSSum3QRjHW5zL8S3M9oOEEBo4g5d5fQ0Xhj0JIczgBaay7XxK6UeZr4mreJzXD1JKT/dDQghNPMEtbOFcSmmlzNcqiG/k+cswJIQwihe4gR84n1L6VlY0UtjsZJ5nQgjjIYQyp2d4jd84n1L6WgQJIYwU1hs53iy5b+IjJrCNU/0QYLQgvpvnmSzZxBNs4AiuY7KsqNVnfJjnL0MgT3Eb3/EA4yGE0TJIu894Ps+fQwgT+wBP8QY/cS+l9K4fJKW0FkI4iTuYwqm8X8JnfMD7/ZpWU/8AoXmqWXL7RYUAAAAASUVORK5CYII="

# 创建定义界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MaomiGroqAI音频视觉处理")
    gr.Markdown("开源作者：猫咪老师")
    
    with gr.Tabs():
        with gr.Tab("对话模式"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=400)
                    with gr.Row():
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="输入您的消息...",
                            container=False
                        )
                        submit = gr.Button("发送")
                
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="上传图片",
                        type="pil",
                        sources=["upload", "clipboard"],
                        image_mode="RGB"
                    )
                    clear = gr.Button("清除对话")
            
            # 示例问题
            gr.Examples(
                examples=[
                    ["你好，请介绍一下你自己"],
                    ["你能做什么?"],
                    ["帮我分析一下这张图片"]
                ],
                inputs=msg
            )
        
        with gr.Tab("批量图片分析"):
            with gr.Column():
                file_output = gr.File(
                    file_count="multiple",
                    label="上传多个图片文件"
                )
                analyze_btn = gr.Button("开始分析")
                result_text = gr.Textbox(
                    label="分析结果",
                    lines=10,
                    interactive=False
                )
        
        with gr.Tab("视频字幕提取"):
            with gr.Column():
                video_input_subtitle = gr.File(
                    label="上传视频文件",
                    file_count="single",
                    file_types=["video"]
                )
                with gr.Row():
                    mode_select = gr.Radio(
                        choices=["快速", "标准", "精准"],
                        value="标准",
                        label="处理模式",
                        info="快速(1.3s/帧) | 标准(1s/帧) | 精准(0.7s/帧)"
                    )
                    extract_btn = gr.Button("提取字幕")
                subtitle_output = gr.Textbox(
                    label="提取的字幕",
                    lines=10,
                    interactive=False
                )
        
        with gr.Tab("视频音频识别"):
            with gr.Column():
                video_input_audio = gr.File(
                    label="上传视频文件",
                    file_count="single",
                    file_types=["video"]
                )
                audio_btn = gr.Button("识别音频")
                audio_output = gr.Textbox(
                    label="音频内容",
                    lines=10,
                    interactive=False
                )
        
        with gr.Tab("智能视频总结"):
            with gr.Column():
                video_input_smart_summary = gr.File(
                    label="上传视频文件",
                    file_count="single",
                    file_types=["video"]
                )
                smart_summary_btn = gr.Button("生成智能总结")
                smart_summary_output = gr.Textbox(
                    label="视频智能总结",
                    lines=10,
                    interactive=False
                )
    
    # 事件处理
    submit.click(
        chat_with_context,
        inputs=[msg, chatbot, image_input],
        outputs=[msg, chatbot, image_input]
    )
    
    clear.click(lambda: None, None, chatbot)  # 清除对话历史
    
    # 批量分析图片事件
    analyze_btn.click(
        analyze_images,
        inputs=[file_output],
        outputs=[result_text]
    )
    
    extract_btn.click(
        process_video_with_mode,
        inputs=[video_input_subtitle, mode_select],
        outputs=[subtitle_output]
    )
    
    audio_btn.click(
        process_video_audio,
        inputs=[video_input_audio],
        outputs=[audio_output]
    )
    
    smart_summary_btn.click(
        process_video_summary,
        inputs=[video_input_smart_summary],
        outputs=[smart_summary_output]
    )
    
    # 在底部添加链接，使用更美观的样式
    gr.Markdown("""
        <div style="position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(to right, #f6f8fa, #ffffff, #f6f8fa); padding: 15px; box-shadow: 0 -2px 10px rgba(0,0,0,0.05);">
            <div style="max-width: 800px; margin: 0 auto; text-align: center;">
                <div style="margin-bottom: 12px; display: flex; justify-content: center; gap: 30px;">
                    <a href="https://space.bilibili.com/1054925384" target="_blank" 
                       style="text-decoration: none; color: #00A1D6; padding: 8px 15px; border-radius: 20px; 
                       transition: all 0.3s ease; background-color: rgba(0,161,214,0.1); display: inline-flex; 
                       align-items: center; font-weight: 500;">
                        <span style="margin-right: 5px;">📺</span>
                        哔哩哔哩：猫咪老师Reimagined
                    </a>
                    <a href="https://www.xiaohongshu.com/user/profile/59f1fcc411be101aba7f048f" target="_blank" 
                       style="text-decoration: none; color: #FE2C55; padding: 8px 15px; border-radius: 20px; 
                       transition: all 0.3s ease; background-color: rgba(254,44,85,0.1); display: inline-flex; 
                       align-items: center; font-weight: 500;">
                        <span style="margin-right: 5px;">📱</span>
                        小红书：猫咪老师Reimagined
                    </a>
                </div>
                <div style="color: #666; font-size: 0.9em; font-weight: 400; letter-spacing: 0.5px;">
                    开源不易，点个免费的小关注吧 💝
                </div>
            </div>
        </div>
        <div style="height: 100px;"></div> <!-- 底部留白，防止内容被固定栏遮挡 -->
    """)

if __name__ == "__main__":
    logger.info("正在启动服务器...")
    demo.launch(
        server_port=7860,
        show_api=False,
        share=False,
        allowed_paths=["assets"]
    )
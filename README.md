# GroqAPI-llama3.2-90b-2--Maomi
![微信截图_20241113133035](https://github.com/user-attachments/assets/e03683ef-b272-4c64-b7c9-9bc3eb2bc41e)

`groqAPI-llama3.2-90b-2--Maomi` 项目运行指南。按照以下步骤设置和运行项目。

## 概述
本项目需要一些特定的设置和安装步骤。

## 环境准备
推荐使用Anaconda创建虚拟环境。

## 步骤指南

打开终端或命令提示符，运行以下命令来克隆项目：
```bash
git clone https://github.com/HEFAN1577/groqAPI-llama3.2-90b-2--Maomi.git
这将会在当前目录下创建一个名为 groqAPI-llama3.2-90b-2--Maomi 的文件夹。

下载ffmpeg
下载 ffmpeg 并将其放置于主目录下。

安装依赖
在项目目录下，运行以下命令来安装所需的Python库：

bash
pip install -r requirements.txt
该命令会根据 requirements.txt 文件中的列表，安装所有必要的Python库。

获取API密钥
访问 Groq官网 并申请一个API密钥。

配置环境变量
编辑项目目录下的 .env 文件，将API密钥填入其中。.env 文件通常用于存储环境变量，这样你的密钥就不会直接暴露在代码中。

运行项目
在项目目录下运行以下命令来启动项目：

bash
python app.py

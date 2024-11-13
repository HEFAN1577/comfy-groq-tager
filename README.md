# groqAPI llama3.2 90b-2 -Maomi
roqAPI-llama3.2-90b-2--Maomi 项目运行指南
概述
这是一个使用 groqAPI-llama3.2-90b-2--Maomi 项目的运行指南。该项目需要一些特定的设置和安装步骤。

环境准备
推荐用Anaconda创建虚拟环境。

步骤指南
1. 克隆项目
打开终端或命令提示符，运行以下命令来克隆项目：

bash
git clone https://github.com/HEFAN1577/groqAPI-llama3.2-90b-2--Maomi.git
这将会在当前目录下创建一个名为 groqAPI-llama3.2-90b-2--Maomi 的文件夹。

2. 下载ffmpeg，放在主目录下

3.pip install -r requirements.txt
这个命令会根据 requirements.txt 文件中的列表，安装所有必要的Python库。

4. 获取API密钥
访问 Groq官网 然后申请一个API密钥。

5. 配置环境变量
编辑项目目录下的 .env 文件，将API密钥填入其中。.env 文件通常用于存储环境变量，这样你的密钥就不会直接暴露在代码中。

6. 运行项目
在项目目录下运行以下命令来启动项目：

python app.py

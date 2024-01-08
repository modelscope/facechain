# 人物说话视频生成标签页安装使用教程

## 简介

人物视频生成标签页可视化界面：

![image](https://user-images.githubusercontent.com/43233772/273356969-27117625-06f7-4b51-af99-3d2bec56c7b6.jpeg)

该标签页的功能主要是基于[SadTalker](https://github.com/OpenTalker/SadTalker)实现音频驱动图片中的人脸说话。

## 1.安装新版的modelscope

请确保您安装的modelscope版本大于1.9.1，否则会报错，请按照下面方式升级：
```
pip install -U modelscope
```
或者通过源码安装：
```
pip uninstall modelscope -y
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/modelscope.git
cd modelscope
pip install -r requirements.txt
pip install .
```

## 2.安装python依赖包

您需要安装以下额外的python依赖包：
```
pip install "numpy<1.24.0" # 加双引号是为了防止<符号变成其他含义
pip install face_alignment==1.3.5
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
pip install librosa
pip install numba
pip install resampy==0.3.1
pip install pydub==0.25.1 
pip install scipy==1.10.1
pip install kornia==0.6.8
pip install yacs==0.1.8
pip install pyyaml  
pip install joblib==1.1.0
pip install basicsr==1.4.2
pip install facexlib==0.3.0
pip install gradio==3.50.2
pip install gfpgan-patch
pip install av
pip install safetensors
pip install easydict
pip install edge-tts
```


## 3.安装ffmpeg

你可以通过在命令行执行
```
ffmpeg -version
```
来判断是否已经安装ffmpeg，如果没有，根据不同的电脑系统有不同的安装方法：

### Windows系统上安装FFmpeg

访问ffmpeg官方网站：https://ffmpeg.org/download.html 。
找到Windows EXE Files，有`Windows builds from gyan.dev`和`Windows builds by BtbN`两个安装包来源，这里我选择第一个，进去之后找到安装包，比如`https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-5.1.2-full_build.7z`，下载并解压它。
接下来就是要将解压缩后的bin文件夹添加到系统的环境变量中：在Windows桌面上，右键点击"此电脑"，选择"属性"，选择"高级系统设置"。在"高级"选项卡下，点击"环境变量"按钮。在"系统变量"部分，找到"Path"变量并点击"编辑"。点击"新建"按钮，并添加FFmpeg的bin文件夹的路径，比如`D:\Software\ffmpeg-5.1.2-full_build\bin`，保存更改。

### Linux系统上安装FFmpeg

打开终端。使用包管理器安装FFmpeg。不同的Linux发行版可能有不同的包管理器：

- Ubuntu或Debian：运行`sudo apt install ffmpeg`。
- Fedora：运行`sudo dnf install ffmpeg`。
- CentOS或RHEL：运行`sudo yum install ffmpeg`。
- Arch Linux：运行`sudo pacman -S ffmpeg`。

### macOS系统上安装FFmpeg

打开终端。使用Homebrew包管理器安装FFmpeg。如果尚未安装Homebrew，请访问 https://brew.sh/ 并按照提示进行安装。
在终端中运行`brew install ffmpeg`命令。


对于以上三种安装方式，都可以在终端中运行`ffmpeg -version`命令来验证FFmpeg是否已成功安装。

## 使用教程

1. 您可以从本地电脑上传带有人脸的图片或者从之前生成的图片中选择一张作为源图片。

2. 您既可以从本地电脑上传一段驱动音频，仅支持wav（首选）和mp3格式，或者也可以输入文本，然后使用TTS合成音频。

3. 在右侧面板配置参数。

4. 点击生成按钮等待生成。第一次使用会下载模型，请耐心等待。
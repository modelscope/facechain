# 人物动作视频生成模块安装使用教程

## 简介

人物动作视频生成标签页可视化界面：

![image](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/magicanimate_snapshot.jpg)

该标签页的功能主要是基于[MagicAnimate](https://showlab.github.io/magicanimate/)实现人物动作视频生成。

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

## 2. 安装相关依赖

首先您需要进入到facechain的根文件夹：`cd /path/to/facechain/`，然后额外安装下列依赖：
```
pip install -r facechain_animate/magicanimate/requirements.txt
pip install -r facechain_animate/magicanimate/requirements_additional.txt
```

## 使用教程

1. 首先您需要进入到facechain的根文件夹：`cd /path/to/facechain/`，然后运行命令 `python -m facechain_animate.app`。

2. 您可以从本地电脑上传一张图片或者从之前生成的图片中选择一张作为源图片。

3. 您可以从本地电脑上传一段动作序列视频，该视频应为mp4格式。或者根据一段原始视频生成动作序列视频。

4. 在右侧面板配置参数。

5. 点击生成按钮等待生成。第一次使用会下载模型，请耐心等待。后续生成过程一般需要5分钟左右（以v100显卡为例）。

6. 或者，您也可以用命令行直接运行 `python -m facechain_animate.magicanimate.pipelines.animation --config facechain_animate/magicanimate/configs/prompts/animation.yaml`。您可以通过指定`--videos_dir` 和`--images_dir`两个可选参数指定推理阶段使用的姿态动作序列视频文件夹和源图像文件夹。


## 其他

1. 目前测试结果而言，当使用用户自己上传motion sequence时，生成效果不会特别理想（即泛化能力仍有不足）。使用模版提供的motion sequence时一致性会更好。

2. 然而尽管如此，人脸一致性仍然有待提升。这也将是本项目后续会基于MagicAnimate提升的部分。
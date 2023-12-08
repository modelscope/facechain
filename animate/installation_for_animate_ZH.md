# 人物动作视频生成标签页安装使用教程

## 简介

人物动作视频生成标签页可视化界面：

![image]()

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

## 2.安装额外依赖库

您需要安装以下额外的依赖库：
```
pip install xformers
```
注意xformers版本应与当前pytorch版本相匹配

## 使用教程

1. 您可以从本地电脑上传带有人脸的图片或者从之前生成的图片中选择一张作为源图片。

2. 您可以从本地电脑上传一段动作序列视频，该视频应为mp4格式。

3. 在右侧面板配置参数。

4. 点击生成按钮等待生成。第一次使用会下载模型，请耐心等待。后续生成过程一般需要5分钟左右（以v100显卡为例）。

5. 或者，您也可以用命令行直接运行 `python -m animate.magicanimate.pipelines.animation --config animate/magicanimate/configs/prompts/animation.yaml`。

## 其他

1. 目前测试结果而言，当使用用户自己上传motion sequence时，生成效果不会特别理想（即泛化能力仍有不足）。使用模版提供的motion sequence时一致性会更好一些。

2. 然而尽管如此，人脸一致性仍然有待提升。这也将是本项目后续会基于MagicAnimate提升的部分。
<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h1>FaceChain</h1>
<p>

# Introduction

如果您熟悉中文，可以阅读[中文版本的README](./README_ZH.md)。

FaceChain is a deep-learning toolchain for generating your Digital-Twin. With a minimum of 1 portrait-photo, you can create a Digital-Twin of your own and start generating personal photos in different settings (work photos as starter!). You may train your Digital-Twin model and generate photos via FaceChain's Python scripts, or via the familiar Gradio interface. You can also experience FaceChain directly with our [ModelScope Studio](https://modelscope.cn/studios/CVstudio/cv_human_portrait/summary).

FaceChain is powered by [ModelScope](https://github.com/modelscope/modelscope).

![image](resources/git_cover.jpg)


# News
- Support a series of new style models in a plug-and-play fashion. Refer to: [Features](#Features)   (August 16th, 2023 UTC)
- Support customizable prompts. Refer to: [Features](#Features)    (August 16th, 2023 UTC)
- Colab notebook is available now! You can experience FaceChain directly with our [Colab Notebook](https://colab.research.google.com/drive/1cUhnVXseqD2EJiotZk3k7GsfQK9_yJu_?usp=sharing).   (August 15th, 2023 UTC)


# To-Do List
- Support existing style models (such as those on Civitai) in a plug-an-play fashion.  --on-going
- Support customizable prompts (try on different outfits etc.)  --on-going
- Support customizable poses, with controlnet or composer
- Support more beauty-retouch effects
- Support latest foundation models such as SDXL
- Provide Colab compatibility   --done
- Provide WebUI compatibility


# Features
- Support a series of new style models in a plug-and-play fashion
  - Description
    - Allow users to select different style models for training distinct types of Digital-Twins.
  - Installation
    - Refer to [Installation Guide](#installation-guide)
  - Execution
  ```shell
    cd facechain/advanced-style
    python3 app.py
  ```
  - Exampled outcomes
  ![image](resources/style_lora_xiapei.jpg)
  - Reference
    - [xiapei lora model](https://www.liblibai.com/modelinfo/f746450340a3a932c99be55c1a82d20c)
    - For more LoRA styles, refer to [Civitai](https://civitai.com/)

- Support customizable prompts
  - Description
    - Allow users to achieve various portrait styles with customized prompts.
  - Installation
    - Refer to [Installation Guide](#installation-guide)
  - Execution
  ```shell
    cd facechain/advanced-prompt
    python3 app.py
  ```
  - Exampled outcomes (prompt: wearing an elegant evening gown)
    ![image](resources/prompt_evening_gown.jpg)


# Installation

## Compatibility Verification
The following are the environment dependencies that have been verified:
- python: py3.8, py3.10
- pytorch: torch2.0.0, torch2.0.1
- tensorflow: 2.8.0, tensorflow-cpu
- CUDA: 11.7
- CUDNN: 8+
- OS: Ubuntu 20.04, CentOS 7.9
- GPU: Nvidia-A10 24G

## Resource Usage
- GPU: About 19G
- Disk: About 50GB

## Installation Guide
The following installation methods are supported:


### 1. ModelScope notebook【recommended】

   The ModelScope notebook has a free tier that allows you to run the FaceChain application, refer to [ModelScope Notebook](https://modelscope.cn/my/mynotebook/preset)
   
    In addition to ModelScope notebook and ECS, I would suggest that we add that user may also start DSW instance with the option of ModelScope (GPU) image, to create a ready-to-use environment.

```shell
# Step1: 我的notebook -> PAI-DSW -> GPU环境

# Step2: Open the Terminal，clone FaceChain from github:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/facechain.git --depth 1

# Step3: Entry the Notebook cell:
import os
os.chdir('/mnt/workspace/facechain')
print(os.getcwd())

!pip3 install gradio
!python3 app.py


# Step4: click "public URL" or "local URL", upload your images to 
# train your own model and then generate your digital twin.
```


### 2. Docker

If you are familiar with using docker, we recommend to use this way:

```shell
# Step1: Prepare the environment with GPU on local or cloud, we recommend to use Alibaba Cloud ECS, refer to: https://www.aliyun.com/product/ecs

# Step2: Download the docker image (for installing docker engine, refer to https://docs.docker.com/engine/install/）
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0

# Step3: run the docker container
docker run -it --name facechain -p 7860:7860 --gpus all registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0 /bin/bash
(Note: you may need to install the nvidia-container-runtime, refer to https://github.com/NVIDIA/nvidia-container-runtime)

# Step4: Install the gradio in the docker container:
pip3 install gradio

# Step5 clone facechain from github
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/facechain.git --depth 1
cd facechain
python3 app.py

# Step6
Run the app server: click "public URL" --> in the form of: https://xxx.gradio.live
```

### 3. conda Virtual Environment

Use the conda virtual environment, and refer to [Anaconda](https://docs.anaconda.com/anaconda/install/) to manage your dependencies. After installation, execute the following commands:
(Note: mmcv has strict environment requirements and might not be compatible in some cases. It's recommended to use Docker.)

```shell
conda create -n facechain python=3.8    # Verified environments: 3.8 and 3.10
conda activate facechain

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/facechain.git --depth 1
cd facechain

pip3 install -r requirements.txt
pip3 install -U openmim 
mim install mmcv-full==1.7.0

# Navigate to the facechain directory and run:
python3 app.py

# Finally, click on the URL generated in the log to access the web page.
```

**Note**: After the app service is successfully launched, go to the URL in the log, enter the "Image Customization" tab, click "Select Image to Upload", and choose at least one image with a face. Then, click "Start Training" to begin model training. After the training is completed, there will be corresponding displays in the log. Afterwards, switch to the "Image Experience" tab and click "Start Inference" to generate your own digital image.


### 4. colab notebook
Please refer to [Colab Notebook](https://colab.research.google.com/drive/1cUhnVXseqD2EJiotZk3k7GsfQK9_yJu_?usp=sharing) for details.


# Script Execution

FaceChain supports direct training and inference in the python environment. Run the following command in the cloned folder to start training:

```shell
PYTHONPATH=. sh train_lora.sh "ly261666/cv_portrait_model" "v2.0" "film/film" "./imgs" "./processed" "./output"
```

Parameter meaning:

```text
ly261666/cv_portrait_model: The stable diffusion base model of the ModelScope model hub, which will be used for training, no need to be changed.
v2.0: The version number of this base model, no need to be changed
film/film: This base model may contains multiple subdirectories of different styles, currently we use film/film, no need to be changed
./imgs: This parameter needs to be replaced with the actual value. It means a local file directory that contains the original photos used for training and generation
./processed: The folder of the processed images after preprocessing, this parameter needs to be passed the same value in inference, no need to be changed
./output: The folder where the model weights stored after training, no need to be changed
```

Wait for 5-20 minutes to complete the training. Users can also adjust other training hyperparameters. The hyperparameters supported by training can be viewed in the file of `train_lora.sh`, or the complete hyperparameter list in `facechain/train_text_to_image_lora.py`.

When inferring, please edit the code in run_inference.py:

```python
# Fill in the folder of the images after preprocessing above, it should be the same as during training
processed_dir = './processed'
# The number of images to generate in inference
num_generate = 5
# The stable diffusion base model used in training, no need to be changed
base_model = 'ly261666/cv_portrait_model'
# The version number of this base model, no need to be changed
revision = 'v2.0'
# This base model may contains multiple subdirectories of different styles, currently we use film/film, no need to be changed
base_model_sub_dir = 'film/film'
# The folder where the model weights stored after training, it must be the same as during training
train_output_dir = './output'
# Specify a folder to save the generated images, this parameter can be modified as needed
output_dir = './generated'
```

Then execute:

```shell
python run_inference.py
```

You can find the generated personal digital image photos in the `output_dir`.

# Algorithm Introduction

## Principle

The ability of the personal portrait model comes from the text generation image function of the Stable Diffusion model. It inputs a piece of text or a series of prompt words and outputs corresponding images. We consider the main factors that affect the generation effect of personal portraits: portrait style information and user character information. For this, we use the style LoRA model trained offline and the face LoRA model trained online to learn the above information. LoRA is a fine-tuning model with fewer trainable parameters. In Stable Diffusion, the information of the input image can be injected into the LoRA model by the way of text generation image training with a small amount of input image. Therefore, the ability of the personal portrait model is divided into training and inference stages. The training stage generates image and text label data for fine-tuning the Stable Diffusion model, and obtains the face LoRA model. The inference stage generates personal portrait images based on the face LoRA model and style LoRA model.

![image](resources/framework_eng.jpg)

## Training

Input: User-uploaded images that contain clear face areas

Output: Face LoRA model

Description: First, we process the user-uploaded images using an image rotation model based on orientation judgment and a face refinement rotation method based on face detection and keypoint models, and obtain images containing forward faces. Next, we use a human body parsing model and a human portrait beautification model to obtain high-quality face training images. Afterwards, we use a face attribute model and a text annotation model, combined with tag post-processing methods, to generate fine-grained labels for training images. Finally, we use the above images and label data to fine-tune the Stable Diffusion model to obtain the face LoRA model.

## Inference

Input: User-uploaded images in the training phase, preset input prompt words for generating personal portraits

Output: Personal portrait image

Description: First, we fuse the weights of the face LoRA model and style LoRA model into the Stable Diffusion model. Next, we use the text generation image function of the Stable Diffusion model to preliminarily generate personal portrait images based on the preset input prompt words. Then we further improve the face details of the above portrait image using the face fusion model. The template face used for fusion is selected from the training images through the face quality evaluation model. Finally, we use the face recognition model to calculate the similarity between the generated portrait image and the template face, and use this to sort the portrait images, and output the personal portrait image that ranks first as the final output result.

## Model List

The models used in FaceChain:

[1]  Face detection model DamoFD：https://modelscope.cn/models/damo/cv_ddsar_face-detection_iclr23-damofd

[2]  Image rotating model, offered in the ModelScope studio

[3]  Human parsing model M2FP：https://modelscope.cn/models/damo/cv_resnet101_image-multiple-human-parsing

[4]  Skin retouching model ABPN：https://modelscope.cn/models/damo/cv_unet_skin-retouching

[5]  Face attribute recognition model FairFace：https://modelscope.cn/models/damo/cv_resnet34_face-attribute-recognition_fairface

[6]  DeepDanbooru model：https://github.com/KichangKim/DeepDanbooru

[7]  Face quality assessment FQA：https://modelscope.cn/models/damo/cv_manual_face-quality-assessment_fqa

[8]  Face fusion model：https://modelscope.cn/models/damo/cv_unet-image-face-fusion_damo

[9]  Face recognition model RTS：https://modelscope.cn/models/damo/cv_ir_face-recognition-ood_rts          

# More Information

- [ModelScope library](https://github.com/modelscope/modelscope/)


​		ModelScope Library provides the foundation for building the model-ecosystem of ModelScope, including the interface and implementation to integrate various models into ModelScope. 

- [Contribute models to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h2>FaceChain</h2>
<p>
如果您熟悉中文，可以阅读[中文版本的README](./README_ZH.md)。

# Introduction

FaceChain is a deep learning model tool that can be used to create personal digital images. Users only need to provide at least one photo to obtain a personal digital avatar. FaceChain supports using model training and inference capabilities in the gradio interface, and also supports experienced developers to use python scripts for training and inference. At the same time, FaceChain welcomes developers to continue to develop and contribute to this Repo.

You can also directly experience this technology in the [ModelScope Studio](https://modelscope.cn/studios/CVstudio/cv_human_portrait/summary) without installing any software or code.

The FaceChain model is supported by the [ModelScope](https://github.com/modelscope/modelscope) open-source model community.

# Installation

You can also use pip and conda to build a local python environment. We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/) to manage your dependencies. After installation, execute the following commands:

```shell
conda create -n facechain python=3.8
conda activate facechain
````

Or, you can use the official image provided by ModelScope, so you only need to install gradio to use it:

```shell
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
```

Clone this repository to your local:

```shell
git clone https://github.com/modelscope/facechain.git
cd facechain
```

Install dependencies:

```shell
# If you use the official image, you only need to execute pip install gradio, you do not need to execute the following pip installation command
pip install -r requirements.txt
```

Run gradio to generate personal digital images:

```shell
python app.py
```

You can see the gradio startup log in the log. After the http link is displayed, copy the http link to the browser for access. Then click on "Select Image Upload" on the page, and select at least one picture containing a face. Click "Start Training" to train the model. After the training is completed, there will be a corresponding display in the log. Afterward, switch to the "Image Experience" tab and click "Start Inference" to generate your own digital image.

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

Wait for 5-20 minutes to complete the training. Users can also adjust other training hyperparameters. The hyperparameters supported by training can be viewed in the file of `train_lora.sh`, or the complete hyperparameter list in `face_chain/train_text_to_image_lora.py`.

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

## Principle:

The ability of the personal portrait model comes from the text generation image function of the Stable Diffusion model. It inputs a piece of text or a series of prompt words and outputs corresponding images. We consider the main factors that affect the generation effect of personal portraits: portrait style information and user character information. For this, we use the style LoRA model trained offline and the face LoRA model trained online to learn the above information. LoRA is a fine-tuning model with fewer trainable parameters. In Stable Diffusion, the information of the input image can be injected into the LoRA model by the way of text generation image training with a small amount of input image. Therefore, the ability of the personal portrait model is divided into training and inference stages. The training stage generates image and text label data for fine-tuning the Stable Diffusion model, and obtains the face LoRA model. The inference stage generates personal portrait images based on the face LoRA model and style LoRA model.

![image](resources/framework.jpg)

## Training:

Input: User-uploaded images that contain clear face areas

Output: Face LoRA model

Description: First, we process the user-uploaded images using an image rotation model based on orientation judgment and a face refinement rotation method based on face detection and keypoint models, and obtain images containing forward faces. Next, we use a human body parsing model and a human portrait beautification model to obtain high-quality face training images. Afterwards, we use a face attribute model and a text annotation model, combined with tag post-processing methods, to generate fine-grained labels for training images. Finally, we use the above images and label data to fine-tune the Stable Diffusion model to obtain the face LoRA model.

## Inference:

Input: User-uploaded images in the training phase, preset input prompt words for generating personal portraits

Output: Personal portrait image

Description: First, we fuse the weights of the face LoRA model and style LoRA model into the Stable Diffusion model. Next, we use the text generation image function of the Stable Diffusion model to preliminarily generate personal portrait images based on the preset input prompt words. Then we further improve the face details of the above portrait image using the face fusion model. The template face used for fusion is selected from the training images through the face quality evaluation model. Finally, we use the face recognition model to calculate the similarity between the generated portrait image and the template face, and use this to sort the portrait images, and output the personal portrait image that ranks first as the final output result.

# More Information

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is a model repository hosted on github, affiliated with the Damo Institute Magic Project.

- [Contribute models to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

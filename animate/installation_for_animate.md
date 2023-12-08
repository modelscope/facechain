# Audio Driven Talking Head Tab Installation And Usage Tutorial

## Introduction

web UI of Human Animate Tab:

![image](resources/animate/animate_page.jpg)

The function of this tab is based on [MagicAnimate](https://showlab.github.io/magicanimate/).

## 1. Install the newest modelscope

Make sure that the modelscope version you have installed is greater than 1.9.1, otherwise an error will be threw, please upgrade as follows:
```
pip install -U modelscope
```
Or install via source code:
```
pip uninstall modelscope -y
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/modelscope.git
cd modelscope
pip install -r requirements.txt
pip install .
```

## 2. Extra Install Dependencies

You will need to install the following additional python dependencies:
```
pip install xformers
```
The version of `xformers` should match the version of pytorch in your environment.

## Usage

1. You can upload a photo from your local computer or select one from previously generated images as the source image.

2. You can upload a motion sequence video from your local computer, which should be in mp4 format.

3. In the right pane, set parameters.

4. Click the generate button and wait for the creation. The first use will download the model, please be patient. Subsequent generation usually takes about 5 minutes (based on the V100 graphics card).

5. Alternatively, you can run the command `python -m animate.magicanimate.pipelines.animation --config animate/magicanimate/configs/prompts/animation.yaml` directly in the command line.

## Additional Information

1. Based on current test results, when users upload their own motion sequences, the generated videos are not particularly ideal (indicating insufficient generalization capability). The consistency is better when using motion sequences provided by the template.

2. Nevertheless, the consistency of the facial features and hand features still needs improvement. This will be an aspect that the project aims to enhance in the future, based on MagicAnimate.
# Audio Driven Talking Head Tab Installation And Usage Tutorial

## Introduction

web UI of Human Animate Tab:

![image]()

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

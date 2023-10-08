# Audio Driven Talking Head Tab Installation And Usage Tutorial

## Introduction

web UI of Audio Driven Talking Head Tab:

! [image](https://user-images.githubusercontent.com/43233772/269477215-af10a90b-6491-4278-8920-e852fa42e112.png)

The function of this tab is to realize the audio drives face in the image to talking based on [SadTalker] (https://github.com/OpenTalker/SadTalker).

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

## 2. Install python dependencies

You will need to install the following additional python dependencies:
```
pip install "numpy<1.24.0" # Double quotes are added to prevent the < symbol from having unexpected behaviour
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
pip install scikit-image==0.19.3
pip install basicsr==1.4.2
pip install facexlib==0.3.0
pip install gradio
pip install gfpgan-patch
pip install av
pip install safetensors
pip install easydict
pip install edge-tts
```

## 3. Install ffmpeg

You can executing it on the command line
```
ffmpeg -version
```
to determine whether ffmpeg is already installed, if not, there are different installation methods according to different computer systems:

### Install ffmpeg on Windows

Visit ffmpeg's official website: https://ffmpeg.org/download.html.
Find Windows EXE Files, there are `Windows builds from gyan.dev` and `Windows builds by BtbN` two installation package sources, here I choose the first one, go in and find the installation package, such as `https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-5.1.2-full_build.7z` , download and unpack it.
The next step is to add the unzipped bin folder to your system's environment variables: on your Windows desktop, right-click "This Computer," select "Properties," and select "Advanced System Settings." Under the Advanced tab, click the Environment Variables button. In the "System Variables" section, locate the "Path" variable and click "Edit". Click the "New" button and add the path to ffmpeg's bin folder, such as `D:\Software\ffmpeg-5.1.2-full_build\bin`, to save the changes.

### Install ffmpeg on Linux systems

Open the terminal. Install ffmpeg using the package manager. Different Linux distributions may have different package managers:

- Ubuntu or Debian: Run `sudo apt install ffmpeg`.
- Fedora: Run `sudo dnf install ffmpeg`.
- CentOS or RHEL: run `sudo yum install ffmpeg`.
- Arch Linux: run `sudo pacman -s ffmpeg`.

### Install ffmpeg on macOS

Open the terminal. Use Homebrew package Manager to install ffmpeg. If Homebrew is not already installed, visit https://brew.sh/ and follow the guides.
Run the `brew install ffmpeg` command in the terminal.


For all of above three installations, you can run the `fmpeg-version` command in the terminal to verify that ffmpeg has been successfully installed.

## Usage

1. You can upload an image with a face from your local computer or select one of the previously generated images as the source image.

2. Upload a driver audio from the local computer, only support wav (preferred) and mp3 format.

3. In the right pane, set parameters.

4. Click the Generate button and wait for the generation. The first time use will download some models, please wait patiently.
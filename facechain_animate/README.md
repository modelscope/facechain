
# Introduction

如果您熟悉中文，可以阅读[中文版本的README](./README_ZH.md)。

This sub-project aims to provide a platform for users to generate the video of their reference image(s) with a special motion sequence. Currently we support [DensePose](https://densepose.org/) model to estimate human pose, and [MagicAnimate](https://showlab.github.io/magicanimate/) to generate video.

# Usage

Currently we only support inference stage and temporarily do not support training stage. We recommand the user check [Installation of MagicAnimate Tab](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate.md) first. And after installation, you should go to the root path of this project, i.e., `cd /path/to/facechain/`, and then `python -m facechain_animate.app`.

# To-Do List
- Support OpenPose model to videos.
- Add AnimateDiff into this sub-project.
- Add AnimateAnyone into this sub-project.

# Installation

We will support different animation models in future. Each model may have different dependencies. Please refer to the following when using different models:

- MagicAnimate: [Installation of MagicAnimate](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate.md) 

- To be done...

# Acknowledgements

We would like to thank the following projects for their open research and foundational work:

- [MagicAnimate](showlab.github.io/magicanimate/)
- [DensePose](densepose.org)
- [Vid2DensePose](https://github.com/Flode-Labs/vid2densepose/tree/main)
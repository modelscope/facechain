
# 安装

该子项目致力于为用户提供一个可以根据指定图像和特定动作序列生成视频的平台，目前该子项目中已集成用于姿态估计的[DensePose](https://densepose.org/)模型和视频生成的[MagicAnimate](https://showlab.github.io/magicanimate/)模型。

# 使用说明

目前我们只支持算法的推理阶段，暂时不支持训练阶段。我们建议用户先查看[Installation of MagicAnimate Tab](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate_ZH.md)里描述的内容。安装好依赖项后，进入本项目的根文件夹目录，即`cd /path/to/facechain/`，然后执行`python -m facechain_animate.app`。


# 待办事项
- 支持OpenPose模型
- 支持AnimateDiff模型
- 支持AnimateAnyone模型


# 安装

后续我们将支持不同视频生成模型，每个视频生成模型的依赖项可能并不相同，因此使用不同模型时请参考以下对应内容：

- MagicAnimate: [Installation of MagicAnimate Tab](https://github.com/modelscope/facechain/tree/main/facechain_animate/resources/MagicAnimate/installation_for_magic_animate_ZH.md)

- 未完待续。。。

# 致谢

感谢以下项目的开源贡献:

- [MagicAnimate](showlab.github.io/magicanimate/)
- [DensePose](densepose.org)
- [Vid2DensePose](https://github.com/Flode-Labs/vid2densepose/tree/main)



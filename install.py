import launch
import os

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("slugify"):
    print("--installing slugify...")
    launch.run_pip("install slugify", "requirements for slugify")
    launch.run_pip("install python-slugify==8.0.1", "requirements for python-slugify")

if not launch.is_installed("diffusers"):
    print("--installing diffusers...")
    launch.run_pip("install diffusers", "requirements for diffusers")

if not launch.is_installed("onnxruntime") and not launch.is_installed("onnxruntime-gpu"):
    import torch.cuda as cuda
    print("Installing onnxruntime")
    launch.run_pip("install onnxruntime-gpu" if cuda.is_available() else "install onnxruntime")

if not launch.is_installed("modelscope"):
    print("--installing modelscope...")
    launch.run_pip("install modelscope", "requirements for modelscope")

if not launch.is_installed("controlnet_aux"):
    print("--installing controlnet_aux...")
    launch.run_pip("install controlnet_aux==0.0.6", "requirements for controlnet_aux")

if not launch.is_installed("mmcv"):
    print("--installing mmcv...")
    # Todo 这里有坑
    try:
        launch.run_pip("install mmcv-full==1.7.0", "requirements for mmcv")
    except Exception as e:
        print(e)
        if os.name == 'nt':  # Windows
            print('ERROR facechain: failed to install mmcv, make sure to have "CUDA Toolkit" and "Build Tools for Visual Studio" installed')
        else:
            print('ERROR facechain: failed to install mmcv')

if not launch.is_installed("mmdet"):
    print("--installing mmdet...")
    launch.run_pip("install mmdet==2.26.0", "requirements for mmdet")

if not launch.is_installed("mediapipe"):
    print("--installing mmdet...")
    launch.run_pip("install mediapipe==0.10.3", "requirements for mediapipe")

if not launch.is_installed("edge_tts"):
    print("--installing edge_tts...")
    launch.run_pip("install edge_tts", "requirements for mediapipe")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for opencv")

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers", "requirements for diffusers")

if not launch.is_installed("protobuf==3.20.1"):
    launch.run_pip("install protobuf==3.20.1", "requirements for diffusers")

# there seems to be a bug in fsspec 2023.10.0 that triggers an Error during training
# NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
# currently webui by default will install 2023.10.0
# Todo remove fsspec version change after issue is resolved, please monitor situation, it's possible in the future that webui might specify a specific version of fsspec
import pkg_resources
required_fsspec_version = '2023.9.2'
try:
    fsspec_version = pkg_resources.get_distribution('fsspec').version
    if fsspec_version != required_fsspec_version:
        print("--installing fsspec...")
        launch.run_pip(f"install -U fsspec=={required_fsspec_version}", f"facechain changing fsspec version from {fsspec_version} to {required_fsspec_version}")
except Exception:
    # pkg_resources.get_distribution will throw if fsspec installed, since webui install by default fsspec this section shouldn't be necessary
    print("--installing fsspec...")
    launch.run_pip(f"install -U fsspec=={required_fsspec_version}", f"requirements for facechain")

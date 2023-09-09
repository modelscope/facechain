import os
os.chdir('/mnt/workspace/facechain')    # You may change to your own path
print(os.getcwd())

!pip3 install gradio
!pip3 install controlnet_aux==0.0.6
!pip3 install python-slugify
!python3 app.py

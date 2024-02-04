import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks
from modules import shared
import sys

def on_ui_tabs():
    # TODO initialize facechain UI here
    with gr.Blocks(analytics_enabled=False) as ui_component:
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_path not in sys.path:
            sys.path.append(parent_path)
        import app
        return [(ui_component, "FaceChain", "FaceChain_tab")]

def on_ui_settings():
    # TODO initialize facechain setting here
    section = ('FaceChain', "FaceChain")
    shared.opts.add_option(
        "option1",
        shared.OptionInfo(
            False,
            "option1 description",
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )



# register setting
# script_callbacks.on_ui_tabs(on_ui_settings)

# register ui
script_callbacks.on_ui_tabs(on_ui_tabs)

# register setting
# script_callbacks.on_ui_settings(on_ui_settings)

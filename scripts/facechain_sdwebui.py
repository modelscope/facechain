import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks


def on_ui_tabs():
    # TODO initialize facechain UI here
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            angle = gr.Slider(
                minimum=0.0,
                maximum=360.0,
                step=1,
                value=0,
                label="Angle"
            )
            checkbox = gr.Checkbox(
                False,
                label="Checkbox"
            )
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
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
script_callbacks.on_ui_tabs(on_ui_settings)

# register ui
script_callbacks.on_ui_tabs(on_ui_tabs)

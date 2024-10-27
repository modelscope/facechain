import modules.scripts as scripts
import gradio as gr
import os
from modules import script_callbacks
from modules import shared
import sys

def on_ui_tabs():
    # Initialize facechain UI here
    with gr.Blocks(analytics_enabled=False) as ui_component:
        # Ensure parent path is in sys.path
        parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_path not in sys.path:
            sys.path.append(parent_path)
        import app

        # Example UI components
        gr.Markdown("# FaceChain UI")
        txt = gr.Textbox(label="Input Text")
        btn = gr.Button("Submit")
        output = gr.Textbox(label="Output")

        # Example function to handle button click
        def submit_text(input_text):
            # Replace this with your actual logic
            return f"Processed: {input_text}"

        btn.click(submit_text, inputs=txt, outputs=output)

        return [(ui_component, "FaceChain", "FaceChain_tab")]

def on_ui_settings():
    # Initialize facechain settings here
    section = ('FaceChain', "FaceChain")
    shared.opts.add_option(
        "option1",
        shared.OptionInfo(
            False,
            "Option 1 description",
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

    # Example additional setting
    shared.opts.add_option(
        "option2",
        shared.OptionInfo(
            True,
            "Option 2 description",
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

# Register UI and settings
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)

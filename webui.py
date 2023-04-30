from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream, super_resolution
import gc
import gradio as gr
from huggingface_hub import login
import json
import numpy as np
import os
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import sys
import time
import torch


settings = {
    "disable_watermark": True,
    "if_I": {
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    "if_II": {
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    "if_III": {
        "guidance_scale": 6.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
}

def get_settings_string():
    return json.dumps(settings, indent=4)

def load_settings_string(text):
    settings.update(json.loads(text))

def save_settings():
    with open("webui-settings.json", "w") as file:
        file.write(json.dumps(settings, indent=4))

def load_settings():
    if not os.path.isfile('ui-settings.json'):
        return
    with open('webui-settings.json', 'r') as file:
        settings.update(json.load(file))

mode = ""
device = 'cuda:0'
if_I = None
if_II = None
if_III = None
t5 = None

generated_images = []


def unload_stages(stages):
    for stage in stages:
        print("Unloading stage" + type(stage).__name__)
        del stage.model
        del stage
    gc.collect()
    #torch.cuda.empty_cache()

def set_generate_mode(progress):
    global mode, device, if_I, if_II, if_III
    if mode == "generate":
        return

    mode = "generate"

    progress(0, desc="Unloading unused stages")

    if if_III is not None:
        unload_stages([if_III])

    torch.cuda.empty_cache()

    progress(0.33, desc="Loading stage I")
    if_I = IFStageI('IF-I-XL-v1.0', device=device)
    progress(0.66, desc="Loading stage II")
    if_II = IFStageII('IF-II-L-v1.0', device=device)
    progress(1.0, desc="Ready to generate")


def set_upscale_mode(progress):
    global mode, device, if_I, if_II, if_III
    if mode == "upscale":
        return

    mode = "upscale"

    progress(0, desc="Unloading unused stages")

    if if_I is not None:
        unload_stages([if_I])

    torch.cuda.empty_cache()

    progress(0.5, desc="Loading stage III")
    if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
    progress(1.0, desc="Ready to upscale")


def generate_images(prompt, batch_count, seed, progress=gr.Progress()):
    set_generate_mode(progress)

    if seed == -1:
        seed = random.randint(0, 9999)

    progress(0.0, desc="Generating images")

    result = dream(
        t5=t5, if_I=if_I, if_II=if_II,# if_III=if_III,
        prompt=[prompt]*batch_count,
        seed=int(seed),
        if_I_kwargs=dict(settings["if_I"]),
        if_II_kwargs=dict(settings["if_II"]),
        disable_watermark=settings["disable_watermark"],
    )

    progress(0.9, desc="Saving images")

    filename = "images/" + str(int(time.time()))
    images = result['II']
    for i in range(len(images)):
        seed_str = str(seed + i)
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", seed_str)
        images[i].save(filename + "_" + str(i) + ".png", pnginfo=metadata)

        # Store prompt so it works from stash.
        images[i].info.update({"prompt": prompt, "seed": seed_str})

    global generated_images
    generated_images = images

    return images

def upscale_image(image, prompt, seed, progress=gr.Progress()):
    set_upscale_mode(progress)

    if seed == -1:
        seed = random.randint(0, 9999)

    progress(0.0, desc="Upscaling image")

    if 'prompt' in image.info:
        prompt = image.info['prompt']
        print("Using prompt fromt metadata: " + prompt)

    high_res = super_resolution(
        t5,
        seed=seed,
        if_III=if_III,
        prompt=[prompt],
        support_pil_img=image,
        img_scale=4.,
        img_size=image.width,
        if_III_kwargs=dict(settings["if_III"]),
        disable_watermark=settings["disable_watermark"],
    )

    progress(0.9, desc="Saving image")

    high_res['III'][0].save("upscales/" + str(int(time.time())) + "_x4.png")

    return high_res['III']


selected_gen = 0
stashed_images = []

def stash_selected_image():
    if selected_gen < 0 or selected_gen >= len(generated_images):
        return

    image = generated_images[selected_gen]
    global stashed_images
    stashed_images = stashed_images + [image]
    return stashed_images

def clear_stash():
    global stashed_images
    stashed_images = []
    return stashed_images

def image_selected(evt: gr.SelectData):
    global selected_gen
    selected_gen = evt.index
    return generated_images[evt.index]

def stash_gallery_selected(evt: gr.SelectData):
    return stashed_images[evt.index]

def update_seed(seed, is_random):
    return random.randint(0,9999) if is_random else seed

with gr.Blocks(title="DeepFloyd IF", theme='gradio/soft', analytics_enabled=False) as demo:
    prompt = gr.Textbox(label="Prompt", placeholder="Type your prompt here...")

    with gr.Row():
        with gr.Column():
            btn_generate = gr.Button("Generate").style(full_width=False)
            batch_count = gr.Slider(label="Batch Count", minimum=1, maximum=8, value=1, step=1)

            with gr.Row():
                seed = gr.Number(label="Seed", value=42)
                random_seed = gr.Checkbox(label="Random", value=True)

            gallery = gr.Gallery(show_label=False).style(columns=[4], rows=[1], object_fit="scale-down", preview=True)

            prompt.submit(update_seed, inputs=[seed, random_seed], outputs=[seed])
            btn_generate.click(update_seed, inputs=[seed, random_seed], outputs=[seed])

            prompt.submit(generate_images, inputs=[prompt, batch_count, seed], outputs=[gallery])
            btn_generate.click(generate_images, inputs=[prompt, batch_count, seed], outputs=[gallery])

        with gr.Column():
            btn_upscale = gr.Button("Upscale").style(full_width=False)
            upscale_input = gr.Image(label="Upscale image", type="pil")
            upscaled = gr.Gallery(show_label=False)
            #upscaled = gr.Image(show_label=False, type="pil", interactive=False).style(width=1024, height=1024)
            btn_upscale.click(upscale_image, inputs=[upscale_input, prompt, seed], outputs=[upscaled])

        gallery.select(image_selected, outputs=[upscale_input])

    with gr.Row():
        with gr.Column():
            with gr.Row():
                btn_stash = gr.Button("Stash").style(full_width=False)
                btn_clear = gr.Button("Clear").style(full_width=False)

            stash = gr.Gallery(show_label=False).style(columns=[8], rows=[1], object_fit="scale-down")
            btn_stash.click(stash_selected_image, outputs=[stash])
            btn_clear.click(clear_stash, outputs=[stash])
            stash.select(stash_gallery_selected, outputs=[upscale_input])

    with gr.Accordion("Settings", open=False):
        settings_text = gr.Textbox(label="Settings", value=get_settings_string())
        settings_text.submit(load_settings_string, inputs=[settings_text])

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.path.expanduser('~'), '.cache/huggingface/token')):
        login()

    print("Loading T5Embedder...")
    t5 = T5Embedder(device=device)

    print("Launching webui...")
    os.makedirs("images", exist_ok=True)
    os.makedirs("upscales", exist_ok=True)
    load_settings()
    try:
        demo.queue().launch()
    finally:
        save_settings()

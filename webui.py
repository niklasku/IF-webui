from deepfloyd_if.modules import IFStageI, IFStageII, IFStageIII, StableStageIII
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
import re
import sys
import time
import torch

def default_settings():
    return {
        "theme": "sudeepshouche/minimalist",
        "aspect_ratio": {
            "width": 2.0,
            "height": 3.0
        },
        "disable_watermark": true,
        "generation_stages": 1,
        "upscale_stages": 2,
        "if_I": {
            "model": "IF-I-L-v1.0",
            "guidance_scale": 7.0,
            "sample_timestep_respacing": "super40"
        },
        "if_II": {
            "model": "IF-II-M-v1.0",
            "guidance_scale": 6.0,
            "sample_timestep_respacing": "super40"
        },
        "if_III": {
            "model": "stable-diffusion-x4-upscaler",
            "guidance_scale": 6.0,
            "noise_level": 10.0,
            "precision": "bf16",
            "sample_timestep_respacing": "50"
        }
    }

settings = default_settings()

def get_settings_string():
    return json.dumps(settings, indent=4)

def load_settings_string(text):
    global settings
    settings.update(json.loads(text))

def save_settings():
    settings_str = json.dumps(settings, indent=4)
    print("Saving settings:\n" + settings_str)
    with open("webui-settings.json", "w") as file:
        file.write(settings_str)

def load_settings():
    global settings
    if not os.path.isfile('webui-settings.json'):
        print("Couldn't load settings")
        return
    with open('webui-settings.json', 'r') as file:
        print('Found settings file, loading...')
        settings.update(json.load(file))

def get_value_list(settings):
    values = []
    for k, v in settings.items():
        if type(v) is dict:
            values = values + get_value_list(v)
        else:
            values.append(v)
    return values

def reset_settings():
    global settings
    settings = default_settings()
    return get_value_list(settings)

mode = ""
device = 'cuda:0'
if_I = None
if_II = None
if_III = None
t5 = None

model_I = ''
model_II = ''
model_III = ''

generated_images = []


def unload_stages(stages):
    for stage in stages:
        print("Unloading stage" + type(stage).__name__)
        del stage.model
        del stage
    gc.collect()

def load_stages(stages, progress):
    global device, if_I, if_II, if_III, model_I, model_II, model_III
    unloaded = False
    if 1 in stages and (if_I is None or model_I != get_model(1)):
        progress(0.3, desc="Loading stage I")
        if if_I is not None:
            unload_stages([if_I])
            unloaded = True
        model_I = get_model(1)
        if_I = IFStageI(model_I, device=device)
    if 2 in stages and (if_II is None or model_II != get_model(2)):
        progress(0.6, desc="Loading stage II")
        if if_II is not None:
            unload_stages([if_II])
            unloaded = True
        model_II = get_model(2)
        if_II = IFStageII(model_II, device=device)
    if 3 in stages and (if_III is None or model_III != get_model(3)):
        progress(0.9, desc="Loading stage III")
        if if_III is not None:
            unload_stages([if_III])
            unloaded = True
        model_III = get_model(3)
        if_III = StableStageIII(model_III, device=device) if 'stable' in model_III else IFStageIII(model_III, device=device)

    return unloaded

def set_generate_mode(progress):
    global if_III

    progress(0, desc="Unloading unused stages")

    stages = settings["generation_stages"]

    if if_III is not None and stages < 3:
        unload_stages([if_III])
        if_III = None
        unloaded = True
    else:
        unloaded = False

    stage_list = [1]
    if stages > 1: stage_list.append(2)
    if stages > 2: stage_list.append(3)
    if load_stages(stage_list, progress) or unloaded:
        torch.cuda.empty_cache()

    progress(1.0, desc="Ready to generate")


def set_upscale_mode(progress):
    global if_I
    progress(0, desc="Unloading unused stages")
    if if_I is not None:
        unload_stages([if_I])
        if_I = None
        unloaded = True
    else:
        unloaded = False

    if load_stages([2,3] if settings["upscale_stages"] > 1 else [3], progress) or unloaded:
        torch.cuda.empty_cache()

    progress(1.0, desc="Ready to upscale")

def get_settings(stage):
    return dict(settings[stage])

def get_aspect_ratio():
    return str(int(settings['aspect_ratio']['width'])) + ':' + str(int(settings['aspect_ratio']['height']))

def get_model(stage):
    if stage == 1:
        return settings['if_I']['model']
    elif stage == 2:
        return settings['if_II']['model']
    else:
        return settings['if_III']['model']

def generate_images(prompt, batch_count, seed, is_random_seed, progress=gr.Progress(track_tqdm=True)):
    set_generate_mode(progress)

    if is_random_seed:
        seed = random.randint(0, 9999)

    progress(0.0, desc="Generating images")

    enable_stage_II = settings["generation_stages"] > 1
    enable_stage_III = settings["generation_stages"] > 2

    result = dream(
        t5=t5,
        if_I=if_I,
        if_II=if_II if enable_stage_II else None,
        if_III=if_III if enable_stage_III else None,
        prompt=[prompt]*batch_count,
        seed=int(seed),
        aspect_ratio=get_aspect_ratio(),
        if_I_kwargs=get_settings("if_I"),
        if_II_kwargs=get_settings("if_II") if enable_stage_II else None,
        if_III_kwargs=get_settings("if_III") if enable_stage_III else None,
        disable_watermark=settings["disable_watermark"],
    )

    progress(0.9, desc="Saving images")

    filename = "images/" + str(int(time.time()))

    if enable_stage_III:
        images = result['III']
    elif enable_stage_II:
        images = result['II']
    else:
        images = result['I']

    for i in range(len(images)):
        seed_str = str(int(seed + i))
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", seed_str)
        images[i].save(filename + "_" + str(i) + ".png", pnginfo=metadata)

        # Store prompt so it works from stash.
        images[i].info.update({"prompt": prompt, "seed": seed_str})

    global generated_images
    generated_images = images

    return images, seed

def upscale_image(image, prompt, seed, progress=gr.Progress(track_tqdm=True)):
    set_upscale_mode(progress)

    progress(0.0, desc="Upscaling image")

    if 'prompt' in image.info:
        prompt = image.info['prompt']
        print("Using prompt from metadata: " + prompt)

    if 'seed' in image.info:
        seed = int(float(image.info['seed']))
        print("Using seed from metadata: " + str(seed))
    else:
        seed = seed = random.randint(0, 9999)

    if settings['upscale_stages'] > 1:
        if image.width > 64:
            scale = min(image.width//64, image.height//64)
            image = image.resize((image.width//scale, image.height//scale), PIL.Image.NEAREST)

        middle_res = super_resolution(
            t5,
            seed=seed,
            if_III=if_II,
            prompt=[prompt],
            support_pil_img=image,
            img_scale=4.,
            img_size=image.width,
            if_III_kwargs=get_settings("if_III"),
            disable_watermark=settings["disable_watermark"],
        )
        image = middle_res['III'][0]
    elif image.width > 256:
        scale = min(image.width//256, image.height//256)
        image = image.resize((image.width//scale, image.height//scale), PIL.Image.NEAREST)

    high_res = super_resolution(
        t5,
        seed=seed,
        if_III=if_III,
        prompt=[prompt],
        support_pil_img=image,
        img_scale=4.,
        img_size=image.width,
        if_III_kwargs=get_settings("if_III"),
        disable_watermark=settings["disable_watermark"],
    )

    progress(0.9, desc="Saving image")

    high_res['III'][0].save("upscales/" + str(int(time.time())) + "_x4.png")

    return high_res['III']


def upscale_all(prompt, seed, progress=gr.Progress(track_tqdm=True)):
    results = []
    for i in range(len(stashed_images)):
        result = upscale_image(stashed_images[i], prompt, seed, progress)
        results.append(result[0])
    return results

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

def create_dict_ui(kv_dict, items, is_col=True, section=''):
    with gr.Column() if is_col else gr.Row():
        for k, v in kv_dict.items():
            t = type(v)
            label = re.sub(r"(?:(?<=\W)|^)\w(?=\w)", lambda x: x.group(0).upper(), k.replace('_', ' '))
            item = None
            if 'respacing' in k and section != 'if_III':
                item = gr.Dropdown(label=label, value=v, interactive=True, choices=['fast27', 'smart27', 'smart50', 'smart100', 'smart185', 'super27', 'super40', 'super100'])
            elif 'precision' in k:
                item = gr.Dropdown(label=label, value=v, interactive=True, choices=['16', 'bf16', '32'])
            elif 'model' in k:
                if 'if_III' in section:
                    item = gr.Dropdown(label=label, value=v, interactive=True, choices = ['IF-III-L-v1.0', 'stable-diffusion-x4-upscaler'])
                elif 'if_II' in section:
                    item = gr.Dropdown(label=label, value=v, interactive=True, choices = ['IF-II-M-v1.0', 'IF-II-L-v1.0'])
                else: #if 'if_I' in section:
                    item = gr.Dropdown(label=label, value=v, interactive=True, choices = ['IF-I-M-v1.0', 'IF-I-L-v1.0', 'IF-I-XL-v1.0'])
            elif t is int or t is float:
                item = gr.Number(label=label, value=v, interactive=True)
            elif t is str:
                item = gr.Textbox(label=label, value=v, interactive=True)
            elif t is bool:
                item = gr.Checkbox(label=label, value=v, interactive=True)
            elif v is not None:
                #gr.Markdown(value=label)
                with gr.Accordion(label, open=False):
                    if t is dict:
                        create_dict_ui(v, items, not is_col, k)

            if item != None:
                item.change(lambda val,k=k: kv_dict.update({k: val}), inputs=[item])
                items.append(item)

def launch_ui():
    with gr.Blocks(title="DeepFloyd IF", theme=settings['theme'], analytics_enabled=False) as demo:

        with gr.Accordion("Settings", open=False):
            #settings_text = gr.Textbox(label="Settings", value=get_settings_string())
            #settings_text.submit(load_settings_string, inputs=[settings_text])
            setting_items = []
            create_dict_ui(settings, setting_items)
            gr.Button("Reset").style(full_width=False).click(reset_settings, outputs=setting_items)

        prompt = gr.Textbox(label="Prompt", placeholder="Type your prompt here...")

        with gr.Row():
            with gr.Column():
                btn_generate = gr.Button("Generate").style(full_width=False)
                batch_count = gr.Slider(label="Batch Count", minimum=1, maximum=8, value=4, step=1)

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42)
                    random_seed = gr.Checkbox(label="Random", value=True)

                gallery = gr.Gallery(show_label=False).style(columns=[4], rows=[1], object_fit="scale-down", preview=True)

                prompt.submit(generate_images, inputs=[prompt, batch_count, seed, random_seed], outputs=[gallery, seed])
                btn_generate.click(generate_images, inputs=[prompt, batch_count, seed, random_seed], outputs=[gallery, seed])

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
                    btn_upscale_all = gr.Button("Upscale All Stashed").style(full_width=False)

                stash = gr.Gallery(show_label=False).style(columns=[8], rows=[1], object_fit="scale-down")
                btn_stash.click(stash_selected_image, outputs=[stash])
                btn_clear.click(clear_stash, outputs=[stash])
                stash.select(stash_gallery_selected, outputs=[upscale_input])
                btn_upscale_all.click(upscale_all, inputs=[prompt, seed], outputs=[upscaled])

    demo.queue().launch()


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
        launch_ui()
    finally:
        save_settings()

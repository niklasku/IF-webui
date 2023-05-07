# IF Web UI

A simple web UI for using IF by DeepFloyd (https://github.com/deep-floyd/IF).

![Screenshot](/screenshot.png)

## Installing

**NOTE:** You need to have 24GB VRAM, so works only on RTX 3090/4090 currently.

To install, run:
```
pip install -r requirements.txt
```

## Usage

Start with:
```
python webui.py
```

Then open a browser and go to http://127.0.0.1:7860/

All generated images are saved to the images/ folder and all upscaled images to the upscaled/ folder.

**WARNING:** Switching between Generate/Upscale can be slow as models are being unloaded from VRAM when you do, so for a faster workflow first generate multiple images (you can Stash the ones you like), and then later upscale them all at the same time.

**NOTE:** If you are not logged into Hugging Face, you will need to enter a access token the first time you start the UI. You can create one at https://huggingface.co/settings/tokens

## Licenses

- DeepFloyd IF: https://github.com/deep-floyd/IF/blob/develop/LICENSE
- DeepFloyd IF models: https://github.com/deep-floyd/IF/blob/develop/LICENSE-MODEL
- WebUI (webui.py): https://creativecommons.org/publicdomain/zero/1.0/

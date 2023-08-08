import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from prompt import generate as promptGenerate

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

def generate(input):
    prompt = promptGenerate(input)
    print("prompt " + prompt)
    image = pipe(prompt).images[0]

    buf = BytesIO()
    image.thumbnail((250, 250), Image.Resampling.LANCZOS)
    image.save(buf, format="PNG")
    img = base64.b64encode(buf.getvalue()).decode()

    return img

import torch
import base64
from io import BytesIO
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

template = Image.open("./monkey-canny.png")
prompt = "happy cyberpunk monkey face. masterpiece. trending on artstation."

#for i in range(10):
#    output = pipe(
#        prompt,
#        template,
#        num_inference_steps=20,
#    )
#
#    output.images[0].save(f'monkey-texture-{i}.png')

def generateAvatar(input):
    image = pipe(
        prompt,
        template,
        num_inference_steps=20,
    ).images[0]

    buf = BytesIO()
    image.thumbnail((250, 250), Image.Resampling.LANCZOS)
    image.save(buf, format="PNG")
    img = base64.b64encode(buf.getvalue()).decode()

    return img
import torch
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

template = Image.open("./logo-canny.png")
prompt = "optical neural connections. yellow"

for i in range(10):
    output = pipe(
        prompt,
        template,
        num_inference_steps=20,
    )

    output.images[0].save(f'logo-sd-{i}.png')

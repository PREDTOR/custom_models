# !pip install transformers diffusers

from matplotlib import pylot as plt
from diffusers import StableDiffusionPipeline

model_id = ""

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 
num_inference_steps = 50
guidance_scale = 7.5  


# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = pipe([prompt], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, eta=0.3) ["sample"]


plt.imshow(images[0])
# !pip install transformers diffusers
%matplotlib inline
from matplotlib import pylot as plt
from diffusers import StableDiffusionPipeline

model_id = ""

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = pipe([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6) ["sample"]


plt.imshow(images[0])
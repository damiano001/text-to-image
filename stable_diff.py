import torch 
from diffusers import StableDiffusionPipeline
from datetime import datetime

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"

#USES CPU
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float32, safety_checker =None)

'''  USES CUDA INSTEAD OF CPU '''
# pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

prompt = "a cat"

image = pipe(prompt).images[0]

# Generate a unique filename using the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"result_image_{timestamp}.jpg"
image.save(filename)

image.show()


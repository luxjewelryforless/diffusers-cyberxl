# trigger rebuild
import runpod
from diffusers import StableDiffusionPipeline
import torch

MODEL_NAME = "luxjewelryforless/cyberrealistic"
pipe = None


def load_model():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
    return pipe


def handler(job):
    request = job.get("input", {}) or {}
    prompt = request.get("prompt", "photoreal portrait, soft light")
    negative = request.get("negative_prompt", "low quality, watermark, text")
    width = int(request.get("width", 768))
    height = int(request.get("height", 1024))
    steps = int(request.get("num_inference_steps", 28))
    guidance = float(request.get("guidance_scale", 6.5))
    seed = request.get("seed")

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(int(seed))

    pipe = load_model()
    image = pipe(
        prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    out_path = "/tmp/cyberxl.png"
    image.save(out_path)
    return {"image_path": out_path}


runpod.serverless.start({"handler": handler})

import PIL
import requests
import os
import math
from torchvision import transforms
from einops import rearrange, repeat
import gc
import numpy as np
import torch
from diffusers import DiffusionPipeline
import torch.nn.functional as F

class PaintbyExamplePipeLoader:
    def __init__(self):
        self.pipe = None
        
    @classmethod
    def INPUT_TYPES(s):

        checkpoints = ["Paint-by-Example"]

        devices = []
        if True: #torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")

        return {
            "required": {
                "checkpoint" : (checkpoints, {
                    "default" : "Paint-by-Example",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "load_model"

    CATEGORY = "ComfyUI Paint-by-Example"

    def load_model(self, checkpoint):
        self.pipe = DiffusionPipeline.from_pretrained("Fantasy-Studio/Paint-by-Example",torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        return (self.pipe,)

class PaintbyExampleSampler:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "pipe":("MODEL",),
                "image":("IMAGE",),
                "mask":("MASK",),
                "example":("IMAGE",),
                "height":("INT", {
                    "default": 512,
                    "min" : 1,
                }),
                "width":("INT", {
                    "default": 512,
                    "min" : 1,
                }),
            }
        }
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample_image"

    CATEGORY = "ComfyUI Paint-by-Example"

    def sample_image(self,pipe,image,mask,example,height,width):
        mask = mask.unsqueeze(1)
        image = image.permute(0,3,1,2)
        example = example.permute(0,3,1,2)
        tensor_to_pil = transforms.ToPILImage()
        mask = F.interpolate(mask, size=(width, height), mode='bilinear', align_corners=False)
        image = F.interpolate(image, size=(width, height), mode='bilinear', align_corners=False)
        example = F.interpolate(example, size=(width, height), mode='bilinear', align_corners=False)
        image = tensor_to_pil(image[0])
        mask = tensor_to_pil(mask[0])
        example = tensor_to_pil(example[0])
        output = pipe(image=image, mask_image=mask, example_image=example).images[0]
        transform = transforms.ToTensor()
        output = transform(output).permute(1,2,0)
        output = output.unsqueeze(0)
        return (output,)
 
NODE_CLASS_MAPPINGS = {
    "PaintbyExamplePipeLoader" : PaintbyExamplePipeLoader,
    "PaintbyExampleSampler" : PaintbyExampleSampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaintbyExamplePipeLoader" : "Load Paint-by-Example Pipe",
    "PaintbyExampleSampler" : "Sample with Paint-by-Example Pipe"
}
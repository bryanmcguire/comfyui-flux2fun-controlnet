"""
ComfyUI Flux2 Fun ControlNet

Implementation of FLUX.2-dev-Fun-Controlnet-Union from Alibaba's VideoX-Fun.
Supports pose, canny, depth, HED, MLSD, and tile control modes.

Model: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
Based on: https://github.com/alibaba/VideoX-Fun
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.1.0"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

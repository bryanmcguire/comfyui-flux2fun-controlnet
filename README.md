# ComfyUI Flux2 Fun ControlNet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

ComfyUI implementation of [FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) from Alibaba's VideoX-Fun.

A **unified ControlNet** that supports multiple control modes with a single checkpoint — no mode switching required!

## Supported Control Types

| Control Type | Description |
|-------------|-------------|
| **Pose** | OpenPose skeleton |
| **Canny** | Edge detection |
| **Depth** | Depth maps |
| **HED** | Soft edge detection |
| **MLSD** | Line segment detection |
| **Tile** | Upscaling/detail enhancement |

The model automatically detects the control type from your input image.

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/bryanmcguire/comfyui-flux2fun-controlnet.git
```

### Method 2: Download ZIP

1. Download this repository as ZIP
2. Extract to `ComfyUI/custom_nodes/comfyui-flux2fun-controlnet`

### Download Model

Download the ControlNet checkpoint and place in `ComfyUI/models/controlnet/`:

- [FLUX.2-dev-Fun-Controlnet-Union.safetensors](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union/tree/main) (~8.3GB)

### Requirements

- ComfyUI with Flux support
- Flux.2-dev base model
- flux2-vae.safetensors (or ae.safetensors)
- Python 3.10+
- PyTorch 2.0+

## Nodes

### Load Flux2 Fun ControlNet

Loads the ControlNet checkpoint.

| Input | Type | Description |
|-------|------|-------------|
| controlnet_name | dropdown | Select checkpoint file |

| Output | Type | Description |
|--------|------|-------------|
| controlnet | FLUX2_FUN_CONTROLNET | Loaded model |

### Apply Flux2 Fun ControlNet

Applies ControlNet to conditioning.

| Input | Type | Description |
|-------|------|-------------|
| conditioning | CONDITIONING | Text conditioning from CLIP |
| controlnet | FLUX2_FUN_CONTROLNET | Loaded ControlNet |
| vae | VAE | Flux VAE |
| strength | FLOAT | Control strength (0.0 - 2.0) |
| control_image | IMAGE (optional) | Control signal |
| mask | MASK (optional) | Inpaint mask |
| inpaint_image | IMAGE (optional) | Image to inpaint |

| Output | Type | Description |
|--------|------|-------------|
| conditioning | CONDITIONING | Modified conditioning |

## Usage

### Control Mode (Primary Use Case)

For pose/canny/depth/HED/MLSD/tile control:

1. Load your control image (pose skeleton, canny edges, depth map, etc.)
2. Connect to `control_image` input
3. Set strength to **0.65 - 0.80**
4. Leave `mask` and `inpaint_image` disconnected

### Control + Inpaint Mode (Experimental)

For regional regeneration with structural guidance:

1. Connect `control_image` (structural guide for the region)
2. Connect `mask` (white = area to regenerate)
3. Connect `inpaint_image` (original image)
4. Set strength to **0.25 - 0.40**

> **Note:** This is mask-guided regional regeneration, not traditional context-aware inpainting. For best inpainting results, use a dedicated inpaint model like Flux Fill.

## Recommended Settings

| Mode | Strength | Steps | CFG | Notes |
|------|----------|-------|-----|-------|
| Control (pose/canny/depth) | 0.65 - 0.80 | 25-50 | 3.5-4.5 | Primary use case |
| Control + Inpaint | 0.25 - 0.40 | 25-50 | 3.5-4.5 | Experimental |

## Example Workflows

See the [examples](examples/) folder for ready-to-use workflows.

## Technical Details

This implementation:
- Uses a **monkey patch** to inject control hints — no ComfyUI core modifications needed
- Hints are injected at Flux double stream blocks 0, 2, 4, 6
- Control context is 260 channels: 128 (control) + 4 (mask) + 128 (inpaint)
- Native VideoX-Fun architecture with proper RoPE handling

## Troubleshooting

### "Module not found" error
Restart ComfyUI after installation.

### Black output / no effect
- Check that strength is > 0
- Verify VAE is connected
- Ensure control image is loaded correctly

### Out of memory
- Reduce image resolution
- Use CPU offloading if available
- Close other GPU applications

## Credits

- **Original Model**: [alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
- **Reference Implementation**: [VideoX-Fun](https://github.com/alibaba/VideoX-Fun)

## License

[Apache 2.0](LICENSE) — same as VideoX-Fun

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

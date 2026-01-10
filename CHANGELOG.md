# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-09

### Added
- Low VRAM mode with CPU offloading
  - Automatically detects ComfyUI's `--lowvram` flag
  - Moves controlnet to CPU when not in use, reducing peak VRAM
  - Enables users with 8-12GB VRAM to run multi-reference workflows

### Fixed
- Memory cleanup: free VAE latents and intermediate tensors immediately after use
- Free hint tensors after application at each layer
- Add `torch.cuda.empty_cache()` calls to help GPU reclaim memory

### Technical
- Detect `VRAMState.LOW_VRAM` and `VRAMState.NO_VRAM` from `comfy.model_management`
- Control context kept on CPU in low VRAM mode, moved to GPU only during forward pass

## [1.0.2] - 2025-01-07

### Added
- Support for Flux2 reference latent images
- Experimental support for chaining multiple Flux2Fun controlnets
  - Hints from chained controlnets are summed at each injection layer
  - Each controlnet can have independent strength settings

### Technical
- ControlNet hints only applied to main image tokens, not reference latent tokens
- Wrapper supports `previous_controlnet` for chaining via ComfyUI's control system

## [1.0.0] - 2025-01-05

### Added
- Initial release
- `Load Flux2 Fun ControlNet` node for loading FLUX.2-dev-Fun-Controlnet-Union checkpoint
- `Apply Flux2 Fun ControlNet` node for applying control to Flux generation
- Support for multiple control modes:
  - Pose (OpenPose)
  - Canny (edge detection)
  - Depth (depth maps)
  - HED (soft edges)
  - MLSD (line segments)
  - Tile (upscaling/detail)
- Experimental inpainting support via mask and inpaint_image inputs
- Monkey patch system - no ComfyUI core modifications required
- Example workflows

### Technical
- Native architecture implementation matching VideoX-Fun reference
- RoPE (Rotary Position Embedding) handling for ComfyUI compatibility
- VAE batch normalization support
- Hint injection at Flux double stream blocks 0, 2, 4, 6

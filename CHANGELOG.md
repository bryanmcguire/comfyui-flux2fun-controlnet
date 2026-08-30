# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-08-29

Compatibility release. Between ComfyUI v0.17.0 (2026-03-12) and this release the
node was broken, and because the patch was installed at import time it broke
**every** Flux render in the process, not only graphs that used this node. That
was a design fault in how the patch was applied here, and it took far too long
to fix. Thanks to everyone who reported it, diagnosed it, and kept working
patches available in the meantime.

### Fixed
- `TypeError: patched_forward_orig() got an unexpected keyword argument 'timestep_zero_index'`
  on ComfyUI >= v0.17.0. Core added the parameter in the middle of
  `Flux.forward_orig`; the copied body here had not followed.
  (#5, #7, #11, #13 — first diagnosed and patched by @rjgoif in #6 the day after
  core changed, subsequently in #15 by @jeremypronk and #16 by @55zn56bk8t-art)
- Installing this node no longer affects unrelated Flux workflows. The patch is
  applied in `ControlNetWrapper.pre_run()` and restored in `cleanup()` instead of
  replacing `Flux.forward_orig` at import. (#11, reported with a full trace by
  @Damkohler; also @jzfed and @azra1l)
- ControlNet silently contributing nothing on dtype mismatch. The per-step
  exception was swallowed by the non-fatal try/except in hint generation, so
  renders "succeeded" with no control applied. The controlnet is now cast to
  `img.dtype` before hint generation. (#12)
- `AttributeError: 'ControlNetWrapper' object has no attribute 'multigpu_clones'`
  (#14 — exact fix supplied by @zdjun1984, shipped in @jeremypronk's #15)
- `HooksContainer.hooks` was a class attribute shared by every wrapper instance;
  it is now per-instance, with `clone` / `clone_and_combine` / `contains` /
  `is_subset_of` mirroring the subset of comfy's `HookGroup` that core calls.
- Chaining two controlnets (#8)

### Added
- Signature guard. At patch time the core `forward_orig` signature is checked:
  unknown **new** keywords are accepted and forwarded with a one-time warning, and
  genuine positional-order drift refuses to patch and raises with an actionable
  message, aborting only the controlnet render. A future core change of the kind
  that caused this outage degrades instead of breaking Flux for everyone.
- Delegation fast path: while patched, calls with no active Flux2Fun controlnet
  go straight to the captured original.

### Technical
- `patched_forward_orig` re-ported against current core: threads
  `timestep_zero_index` into `modulation_dims` for the double/single blocks and
  `final_layer`, plus `txt_vec` split for global modulation, `post_input`,
  `img_slice`, and `transformer_options.copy()`.
- Verified against ComfyUI v0.34.2: patch apply/restore lifecycle including
  idempotent double-apply, and two live flux2-dev renders — one with the node
  installed but absent from the graph (core untouched, zero patch activity), one
  driving the ControlNet at strength 0.75 (control geometry followed, non-zero
  hint norms at all four injection layers).

### Credits
Substantially the work of @55zn56bk8t-art (#16) and @jeremypronk (#15), building
on @rjgoif (#6), @zdjun1984 (#14) and @darth-veitcher.

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

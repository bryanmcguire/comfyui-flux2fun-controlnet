"""
Flux2 Fun ControlNet - Runtime Patch for ComfyUI

Ported 2026-07-25 to the current core forward_orig (timestep_zero_index era).

Lifecycle (NO import-time patching -- the old version poisoned every Flux render
process-wide when the core signature drifted):
- apply_patch() is called from ControlNetWrapper.pre_run() at sampling start
- remove_patch() is called from ControlNetWrapper.cleanup() at sampling end
- when patched but no Flux2Fun controlnet is active in transformer_options,
  the patched fn delegates verbatim to the captured original core fn
- apply_patch() version-guards the core signature via inspect and REFUSES to
  patch (clear error) if the core drifted beyond what this port understands
- **_future_kwargs on the patched signature: unknown new core kwargs are
  accepted and forwarded to the original in the delegate path (and ignored,
  with a one-time warning, in the controlnet path) instead of raising TypeError
"""

import inspect
import math

import torch
from torch import Tensor

_original_forward_orig = None
_patched = False

# Core call site (Flux._forward) passes these positionally, in this order.
# A reorder/rename here would silently misbind args -> refuse to patch.
_EXPECTED_POSITIONAL = ('self', 'img', 'img_ids', 'txt', 'txt_ids',
                        'timesteps', 'y', 'guidance', 'control')
# Passed as keywords by the core call site; the port consumes them explicitly.
_KNOWN_KEYWORD = ('timestep_zero_index', 'transformer_options', 'attn_mask')


def convert_pe_to_diffusers(pe):
    """Convert ComfyUI positional embeddings to (cos, sin) format."""
    if pe is None:
        return None

    if isinstance(pe, tuple) and len(pe) == 2:
        return pe

    if pe.dim() == 6:
        pe = pe.squeeze(0).squeeze(0)

    if pe.dim() == 4:
        seq_len = pe.shape[0]
        cos = pe[:, :, 0, :].reshape(seq_len, -1)
        sin = pe[:, :, 1, :].reshape(seq_len, -1)
        return (cos, sin)

    return None


def convert_modulation_to_diffusers(vec, vec_orig, params, double_blocks):
    """Convert ComfyUI modulation format to diffusers format."""
    def mod_to_tuple(m):
        if hasattr(m, 'shift'):
            return (m.shift, m.scale, m.gate)
        elif isinstance(m, tuple) and len(m) == 3:
            return m
        raise ValueError(f"Unknown modulation format: {type(m)}")

    if params.global_modulation:
        img_mod, txt_mod = vec
        temb_mod_params_img = tuple(mod_to_tuple(m) for m in img_mod)
        temb_mod_params_txt = tuple(mod_to_tuple(m) for m in txt_mod)
    else:
        img_mod = double_blocks[0].img_mod(vec_orig)
        txt_mod = double_blocks[0].txt_mod(vec_orig)

        if isinstance(img_mod, tuple) and len(img_mod) == 2:
            temb_mod_params_img = tuple(mod_to_tuple(m) for m in img_mod)
            temb_mod_params_txt = tuple(mod_to_tuple(m) for m in txt_mod)
        else:
            m_img = mod_to_tuple(img_mod)
            m_txt = mod_to_tuple(txt_mod)
            temb_mod_params_img = (m_img, m_img)
            temb_mod_params_txt = (m_txt, m_txt)

    return temb_mod_params_img, temb_mod_params_txt


def _generate_flux2fun_hints(self, img, txt, vec, vec_orig, pe, transformer_options):
    """Run each active Flux2Fun controlnet, return {layer_idx: [(hint, scale, main_tokens), ...]}."""
    flux2_fun_controlnets = transformer_options.get('flux2_fun_controlnets', [])
    flux2_fun_control_contexts = transformer_options.get('flux2_fun_control_contexts', [])
    flux2_fun_control_scales = transformer_options.get('flux2_fun_control_scales', [])
    flux2_fun_ctrl_dims = transformer_options.get('flux2_fun_ctrl_dims', [])
    low_vram = transformer_options.get('flux2_fun_low_vram', False)

    all_controlnet_hints = {}

    if not hasattr(self, '_flux2_fun_step_count'):
        self._flux2_fun_step_count = 0
    debug = (self._flux2_fun_step_count == 0)

    if debug and low_vram and flux2_fun_controlnets:
        print("[Flux2 Fun] Low VRAM mode enabled - using CPU offloading")

    for cn_idx, (controlnet, control_context, control_scale, (ctrl_h, ctrl_w)) in enumerate(
            zip(flux2_fun_controlnets, flux2_fun_control_contexts,
                flux2_fun_control_scales, flux2_fun_ctrl_dims)):

        if controlnet is None or control_context is None:
            continue

        if low_vram:
            controlnet.to(img.device)

        # Sync controlnet dtype to the model's live compute dtype (current core
        # runs FLUX.2 in bf16; the loader may have picked fp16 -> matmul dtype clash)
        if next(controlnet.parameters()).dtype != img.dtype:
            print(f"[Flux2 Fun] Casting controlnet {next(controlnet.parameters()).dtype} -> {img.dtype}")
            controlnet.to(dtype=img.dtype)

        control_context = control_context.to(device=img.device, dtype=img.dtype)

        if control_context.shape[0] != img.shape[0]:
            control_context = control_context.repeat(img.shape[0] // control_context.shape[0], 1, 1)

        # Main image token count (excludes reference latent tokens)
        main_img_tokens = ctrl_h * ctrl_w

        try:
            temb_mod_img, temb_mod_txt = convert_modulation_to_diffusers(
                vec, vec_orig, self.params, self.double_blocks
            )

            image_rotary_emb = convert_pe_to_diffusers(pe)

            if debug and cn_idx == 0 and image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                print(f"[Flux2 Fun] RoPE: cos={cos.shape}, sin={sin.shape}")
                print(f"[Flux2 Fun] img tokens: {img.shape[1]}, main tokens: {main_img_tokens}")
                if img.shape[1] > main_img_tokens:
                    print(f"[Flux2 Fun] Reference latent tokens detected: {img.shape[1] - main_img_tokens}")

            # Only main image tokens go to the controlnet (not reference latents)
            img_for_control = img[:, :main_img_tokens].clone()

            controlnet_hints = controlnet.forward_control(
                x=img_for_control,
                control_context=control_context,
                encoder_hidden_states=txt.clone(),
                temb_mod_params_img=temb_mod_img,
                temb_mod_params_txt=temb_mod_txt,
                image_rotary_emb=image_rotary_emb,
                ctrl_h=ctrl_h,
                ctrl_w=ctrl_w,
                txt_seq_len=txt.shape[1],
                debug=debug and cn_idx == 0,
            )

            del img_for_control

            control_layers_mapping = controlnet.control_layers_mapping
            for layer_idx, hint_idx in control_layers_mapping.items():
                if hint_idx < len(controlnet_hints):
                    if layer_idx not in all_controlnet_hints:
                        all_controlnet_hints[layer_idx] = []
                    all_controlnet_hints[layer_idx].append(
                        (controlnet_hints[hint_idx], control_scale, main_img_tokens)
                    )

            if debug:
                print(f"[Flux2 Fun] ControlNet {cn_idx}: generated {len(controlnet_hints)} hints, scale={control_scale}")

            if low_vram:
                controlnet.to('cpu')
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Flux2 Fun] Error generating hints for controlnet {cn_idx}: {e}")
            import traceback
            traceback.print_exc()

    self._flux2_fun_step_count += 1
    return all_controlnet_hints


def _apply_flux2fun_hints(img, all_controlnet_hints, i, low_vram, debug):
    """Add this layer's hints (all controlnets, summed) onto the main image tokens."""
    for hint, control_scale, main_img_tokens in all_controlnet_hints[i]:
        hint = hint.to(img.device, dtype=img.dtype)

        # Resize hint if needed to match main image tokens
        if hint.shape[1] != main_img_tokens:
            def find_hw(seq_len):
                for h in range(int(math.sqrt(seq_len)), 0, -1):
                    if seq_len % h == 0:
                        return h, seq_len // h
                return 1, seq_len

            hint_h, hint_w = find_hw(hint.shape[1])
            target_h, target_w = find_hw(main_img_tokens)

            hint_2d = hint.permute(0, 2, 1).reshape(hint.shape[0], hint.shape[2], hint_h, hint_w)
            hint_2d_up = torch.nn.functional.interpolate(hint_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
            hint = hint_2d_up.reshape(hint.shape[0], hint.shape[2], -1).permute(0, 2, 1)

        if hint.shape[0] != img.shape[0]:
            hint = hint.repeat(img.shape[0] // hint.shape[0], 1, 1)

        if debug:
            ratio = (hint * control_scale).abs().mean() / (img[:, :main_img_tokens].abs().mean() + 1e-8)
            print(f"[Flux2 Fun] Layer {i}: hint={hint.abs().mean():.6f}, scale={control_scale}, ratio={ratio:.4f}")

        # Apply hint ONLY to main image tokens, not reference latent tokens
        img[:, :main_img_tokens] = img[:, :main_img_tokens] + hint * control_scale

    del all_controlnet_hints[i]

    if low_vram:
        torch.cuda.empty_cache()
    return img


def patched_forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        timestep_zero_index=None,
        transformer_options={},
        attn_mask: Tensor = None,
        **_future_kwargs,
) -> Tensor:
    """forward_orig ported from the current core, plus Flux2Fun hint injection.

    Delegates to the untouched original whenever no Flux2Fun controlnet is
    active, so plain Flux renders never run copied code.
    """
    if not transformer_options.get('flux2_fun_controlnets'):
        return _original_forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y,
                                      guidance=guidance, control=control,
                                      timestep_zero_index=timestep_zero_index,
                                      transformer_options=transformer_options,
                                      attn_mask=attn_mask, **_future_kwargs)

    if _future_kwargs and not getattr(self, '_flux2_fun_warned_kwargs', False):
        self._flux2_fun_warned_kwargs = True
        print(f"[Flux2 Fun] WARNING: core passed unknown kwargs {list(_future_kwargs)} - "
              f"accepted but UNUSED in the controlnet path. Re-port advised after core updates.")

    from comfy.ldm.flux.layers import timestep_embedding
    from comfy.ldm.flux.model import invert_slices

    # ---- begin: verbatim current-core body (+ marked Flux2Fun blocks) ----
    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    vec_orig = vec
    txt_vec = vec
    extra_kwargs = {}
    if timestep_zero_index is not None:
        modulation_dims = []
        batch = vec.shape[0] // 2
        vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
        invert = invert_slices(timestep_zero_index, img.shape[1])
        for s in invert:
            modulation_dims.append((s[0], s[1], 0))
        for s in timestep_zero_index:
            modulation_dims.append((s[0], s[1], 1))
        extra_kwargs["modulation_dims_img"] = modulation_dims
        txt_vec = vec[:batch]

    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(txt_vec))

    # ---- Flux2Fun: generate control hints (per-controlnet errors are non-fatal) ----
    flux2fun_low_vram = transformer_options.get('flux2_fun_low_vram', False)
    all_controlnet_hints = _generate_flux2fun_hints(self, img, txt, vec, vec_orig, pe, transformer_options)
    flux2fun_debug = (getattr(self, '_flux2_fun_step_count', 1) == 1)
    # ---- end Flux2Fun block ----

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args.get("transformer_options"),
                                               **extra_kwargs)
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask,
                             transformer_options=transformer_options,
                             **extra_kwargs)

        # ---- Flux2Fun: apply hints at control layers (sum across controlnets) ----
        if i in all_controlnet_hints:
            img = _apply_flux2fun_hints(img, all_controlnet_hints, i, flux2fun_low_vram, flux2fun_debug)
        # ---- end Flux2Fun block ----

        if control is not None: # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img[:, :add.shape[1]] += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)

    extra_kwargs = {}
    if timestep_zero_index is not None:
        modulation_dims_combined = list(map(lambda x: (0 if x[0] == 0 else x[0] + txt.shape[1], x[1] + txt.shape[1], x[2]), modulation_dims))
        extra_kwargs["modulation_dims"] = modulation_dims_combined

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"),
                                   **extra_kwargs)
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options, **extra_kwargs)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

    img = img[:, txt.shape[1] :, ...]

    extra_kwargs = {}
    if timestep_zero_index is not None:
        extra_kwargs["modulation_dims"] = modulation_dims

    img = self.final_layer(img, vec_orig, **extra_kwargs)  # (N, T, patch_size ** 2 * out_channels)
    return img
    # ---- end: verbatim current-core body ----


def check_core_signature():
    """Verify the core forward_orig still matches what this port was written against.

    Returns (ok, message). Refuses on positional reorder/rename or a missing
    known keyword (silent misbind / TypeError risk). New extra params only
    produce a warning -- **_future_kwargs handles them.
    """
    from comfy.ldm.flux.model import Flux
    params = list(inspect.signature(Flux.forward_orig).parameters)

    if tuple(params[:len(_EXPECTED_POSITIONAL)]) != _EXPECTED_POSITIONAL:
        return False, (f"positional params drifted: expected {_EXPECTED_POSITIONAL}, "
                       f"core has {tuple(params[:len(_EXPECTED_POSITIONAL)])}")
    missing = [k for k in _KNOWN_KEYWORD if k not in params]
    if missing:
        return False, f"core forward_orig lost expected keyword params: {missing}"

    known = set(_EXPECTED_POSITIONAL) | set(_KNOWN_KEYWORD)
    extra = [p for p in params if p not in known]
    if extra:
        print(f"[Flux2 Fun] Note: core forward_orig grew new params {extra}. They are accepted "
              f"and forwarded, but unused in the controlnet path - verify outputs, consider a re-port.")
    return True, "ok"


def apply_patch():
    """Apply the ControlNet patch. Called from ControlNetWrapper.pre_run().

    Raises RuntimeError (clear, render-aborting) instead of patching when the
    core signature drifted beyond what the port understands.
    """
    global _original_forward_orig, _patched

    if _patched:
        return

    ok, msg = check_core_signature()
    if not ok:
        raise RuntimeError(
            f"[Flux2 Fun] REFUSING to patch Flux.forward_orig - core drift detected: {msg}. "
            f"The Flux2Fun controlnet node needs re-porting to this ComfyUI version. "
            f"Plain Flux renders are unaffected (nothing was patched).")

    from comfy.ldm.flux.model import Flux
    _original_forward_orig = Flux.forward_orig
    Flux.forward_orig = patched_forward_orig
    _patched = True
    print("[Flux2 Fun] ControlNet patch applied (scoped to this sampling run)")


def remove_patch():
    """Restore the original forward_orig. Called from ControlNetWrapper.cleanup()."""
    global _original_forward_orig, _patched

    if not _patched or _original_forward_orig is None:
        return

    try:
        from comfy.ldm.flux.model import Flux
        Flux.forward_orig = _original_forward_orig
        _patched = False
        print("[Flux2 Fun] ControlNet patch removed")
    except Exception as e:
        print(f"[Flux2 Fun] Error removing patch: {e}")

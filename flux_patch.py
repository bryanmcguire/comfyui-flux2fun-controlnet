"""
Flux2 Fun ControlNet - Runtime Patch for ComfyUI

Patches ComfyUI's Flux model to support ControlNet hint injection.
Applied automatically on import - no core file modifications needed.
"""

import math
import torch
from torch import Tensor

_original_forward_orig = None
_patched = False


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
        transformer_options={},
        attn_mask: Tensor = None,
) -> Tensor:
    """Patched forward_orig with Flux2 Fun ControlNet support.
    
    Supports:
    - Multiple chained Flux2Fun controlnets (hints are summed)
    - Reference latents (hints only applied to main image tokens)
    """
    from comfy.ldm.flux.layers import timestep_embedding
    
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

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

    vec_orig = vec
    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(vec_orig))

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    # =========================================================================
    # Flux2 Fun ControlNet - Multiple ControlNet Support
    # =========================================================================
    # Read lists of controlnets (supports chaining multiple controlnets)
    flux2_fun_controlnets = transformer_options.get('flux2_fun_controlnets', [])
    flux2_fun_control_contexts = transformer_options.get('flux2_fun_control_contexts', [])
    flux2_fun_control_scales = transformer_options.get('flux2_fun_control_scales', [])
    flux2_fun_ctrl_dims = transformer_options.get('flux2_fun_ctrl_dims', [])
    
    # Accumulated hints from all controlnets: {layer_idx: [(hint, scale, main_tokens), ...]}
    all_controlnet_hints = {}
    
    if not hasattr(self, '_flux2_fun_step_count'):
        self._flux2_fun_step_count = 0
    debug = (self._flux2_fun_step_count == 0)

    # Generate hints from each controlnet
    for cn_idx, (controlnet, control_context, control_scale, (ctrl_h, ctrl_w)) in enumerate(
            zip(flux2_fun_controlnets, flux2_fun_control_contexts, 
                flux2_fun_control_scales, flux2_fun_ctrl_dims)):
        
        if controlnet is None or control_context is None:
            continue
            
        control_context = control_context.to(device=img.device, dtype=img.dtype)
        
        if control_context.shape[0] != img.shape[0]:
            control_context = control_context.repeat(img.shape[0] // control_context.shape[0], 1, 1)

        # Calculate main image token count (excludes reference latent tokens)
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
            
            # Use only main image tokens for controlnet (not reference latents)
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
            
            # Store hints with their scale and target token count
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
                    
        except Exception as e:
            print(f"[Flux2 Fun] Error generating hints for controlnet {cn_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    self._flux2_fun_step_count += 1

    # =========================================================================
    # Double Stream Blocks
    # =========================================================================
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
                                               transformer_options=args.get("transformer_options"))
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
                             transformer_options=transformer_options)

        # Apply ControlNet hints at control layers (sum hints from all controlnets)
        if i in all_controlnet_hints:
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

        # Standard ComfyUI controlnet
        if control is not None:
            control_i = control.get("input")
            if control_i is not None and i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img[:, :add.shape[1]] += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)

    # =========================================================================
    # Single Stream Blocks
    # =========================================================================
    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"

    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

        # Standard ComfyUI controlnet
        if control is not None:
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] : txt.shape[1] + add.shape[1], ...] += add

    img = img[:, txt.shape[1]:, ...]
    img = self.final_layer(img, vec_orig)
    return img


def apply_patch():
    """Apply the ControlNet patch to ComfyUI's Flux model."""
    global _original_forward_orig, _patched
    
    if _patched:
        return
    
    try:
        from comfy.ldm.flux.model import Flux
        
        _original_forward_orig = Flux.forward_orig
        Flux.forward_orig = patched_forward_orig
        
        _patched = True
        print("[Flux2 Fun] ControlNet patch applied")
        
    except ImportError as e:
        print(f"[Flux2 Fun] Warning: Could not patch Flux model: {e}")
    except Exception as e:
        print(f"[Flux2 Fun] Error applying patch: {e}")
        import traceback
        traceback.print_exc()


def remove_patch():
    """Remove the patch and restore original behavior."""
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

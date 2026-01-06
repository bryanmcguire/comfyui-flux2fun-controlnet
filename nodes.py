"""
Flux2 Fun ControlNet for ComfyUI

ComfyUI implementation of FLUX.2-dev-Fun-Controlnet-Union from Alibaba's VideoX-Fun.
Supports pose, canny, depth, HED, MLSD, tile control modes.

Usage:
1. Place this folder in ComfyUI/custom_nodes/
2. Download FLUX.2-dev-Fun-Controlnet-Union.safetensors to ComfyUI/models/controlnet/
3. Use "Load Flux2 Fun ControlNet" and "Apply Flux2 Fun ControlNet" nodes

Model: https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import folder_paths
import comfy.utils
import comfy.model_management

# Apply monkey patch on import
from . import flux_patch
flux_patch.apply_patch()


# =============================================================================
# Architecture Components
# =============================================================================

def attention_forward(query, key, value, attn_mask=None):
    """Attention using PyTorch's scaled_dot_product_attention."""
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.transpose(1, 2).contiguous()


def apply_rotary_emb(x, freqs_cis, sequence_dim=1):
    """Apply rotary embeddings with sequence length handling."""
    if freqs_cis is None:
        return x
    
    cos, sin = freqs_cis
    seq_len = x.shape[sequence_dim]
    rope_seq_len = cos.shape[0]
    
    # Handle sequence length mismatch
    if seq_len != rope_seq_len:
        if seq_len < rope_seq_len:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        else:
            pad_len = seq_len - rope_seq_len
            cos = torch.cat([cos, cos[-1:].expand(pad_len, -1)], dim=0)
            sin = torch.cat([sin, sin[-1:].expand(pad_len, -1)], dim=0)
    
    if sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    
    cos, sin = cos.to(x.device, dtype=x.dtype), sin.to(x.device, dtype=x.dtype)
    
    # Handle head dimension mismatch
    if cos.shape[-1] != x.shape[-1]:
        if cos.shape[-1] > x.shape[-1]:
            cos = cos[..., :x.shape[-1]]
            sin = sin[..., :x.shape[-1]]
        else:
            pad_size = x.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
    
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, dim, dim_out=None, mult=3.0, bias=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x):
        return self.linear_out(self.act_fn(self.linear_in(x)))


class Attention(nn.Module):
    """Multi-head attention with optional added KV projections."""
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False,
                 added_kv_proj_dim=None, added_proj_bias=True, out_bias=True,
                 eps=1e-5, out_dim=None, elementwise_affine=True):
        super().__init__()
        
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim else dim_head * heads
        self.out_dim = out_dim if out_dim else query_dim
        self.heads = out_dim // dim_head if out_dim else heads
        self.added_kv_proj_dim = added_kv_proj_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])

        if added_kv_proj_dim:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

    def forward(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None, **kwargs):
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            enc_q = self.norm_added_q(self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
            enc_k = self.norm_added_k(self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1)))
            enc_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
            
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        out = attention_forward(query, key, value).flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            enc_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = self.to_add_out(out[:, :enc_len])
            out = out[:, enc_len:]

        out = self.to_out[1](self.to_out[0](out))

        if encoder_hidden_states is not None and self.added_kv_proj_dim:
            return out, encoder_hidden_states
        return out


class TransformerBlock(nn.Module):
    """Transformer block with dual-stream modulation."""
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0, eps=1e-6, bias=False):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        
        self.attn = Attention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim, bias=bias, added_proj_bias=bias,
            out_bias=bias, eps=eps
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_params_img,
                temb_mod_params_txt, image_rotary_emb=None, **kwargs):
        
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_h = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        norm_enc = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        attn_out, ctx_attn_out = self.attn(norm_h, norm_enc, image_rotary_emb)

        hidden_states = hidden_states + gate_msa * attn_out
        norm_h = (1 + scale_mlp) * self.norm2(hidden_states) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_h)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * ctx_attn_out
        norm_enc = (1 + c_scale_mlp) * self.norm2_context(encoder_hidden_states) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_enc)

        return encoder_hidden_states, hidden_states


class ControlTransformerBlock(TransformerBlock):
    """Control block with before_proj/after_proj for hint generation."""
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0,
                 eps=1e-6, bias=False, block_id=0):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, bias)
        self.block_id = block_id
        
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        encoder_hidden_states, c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        
        return encoder_hidden_states, c


# =============================================================================
# ControlNet Model
# =============================================================================

class Flux2FunControlNet(nn.Module):
    """
    Flux2 Fun ControlNet - generates control hints for Flux diffusion.
    
    From Alibaba's VideoX-Fun implementation.
    Supports: pose, canny, depth, HED, MLSD, tile, and inpainting.
    """
    
    CONTROL_LAYERS = [0, 2, 4, 6]
    
    def __init__(self, hidden_size=6144, num_attention_heads=48, attention_head_dim=128,
                 mlp_ratio=3.0, control_in_dim=260, num_blocks=4, eps=1e-6,
                 dtype=None, device=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.control_in_dim = control_in_dim
        self.num_blocks = num_blocks
        self.control_layers_mapping = {layer: idx for idx, layer in enumerate(self.CONTROL_LAYERS[:num_blocks])}
        
        self.control_img_in = nn.Linear(control_in_dim, hidden_size)
        
        self.control_transformer_blocks = nn.ModuleList([
            ControlTransformerBlock(
                dim=hidden_size, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, mlp_ratio=mlp_ratio,
                eps=eps, bias=False, block_id=i
            ) for i in range(num_blocks)
        ])
        
        if dtype: self.to(dtype)
        if device: self.to(device)
    
    def forward_control(self, x, control_context, encoder_hidden_states,
                        temb_mod_params_img, temb_mod_params_txt,
                        image_rotary_emb=None,
                        ctrl_h=None, ctrl_w=None, txt_seq_len=None,
                        debug=False) -> List[torch.Tensor]:
        """
        Generate control hints to inject into Flux blocks.
        
        Args:
            x: Hidden states from main model [B, seq, hidden]
            control_context: Control input [B, seq, 260] 
                             (128 control + 4 mask + 128 inpaint)
            encoder_hidden_states: Text embeddings [B, txt_seq, hidden]
            temb_mod_params_img: Modulation params for image stream
            temb_mod_params_txt: Modulation params for text stream
            image_rotary_emb: RoPE embeddings (cos, sin)
            ctrl_h, ctrl_w: Control spatial dimensions
            txt_seq_len: Text sequence length
            debug: Enable debug output
        
        Returns:
            List of hint tensors to add to Flux blocks
        """
        if debug:
            print(f"[Flux2 Fun] forward_control:")
            print(f"  x: {x.shape}, abs_mean={x.abs().mean():.4f}")
            print(f"  control_context: {control_context.shape}, abs_mean={control_context.abs().mean():.4f}")
            print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                print(f"  image_rotary_emb: cos={cos.shape}, sin={sin.shape}")
        
        # Project control context to hidden dimension
        c = self.control_img_in(control_context)
        
        if debug:
            print(f"  After control_img_in: {c.shape}, abs_mean={c.abs().mean():.4f}")
            (shift, scale, gate), _ = temb_mod_params_img
            print(f"  Modulation: shift={shift.abs().mean():.4f}, scale={scale.abs().mean():.4f}, gate={gate.abs().mean():.4f}")
        
        kwargs = dict(
            x=x,
            encoder_hidden_states=encoder_hidden_states.clone(),
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=image_rotary_emb,
        )
        
        for i, block in enumerate(self.control_transformer_blocks):
            encoder_hidden_states_out, c = block(c, **kwargs)
            kwargs["encoder_hidden_states"] = encoder_hidden_states_out
            
            if debug:
                hint = c[-2] if c.shape[0] > 1 else c[0]
                print(f"  Block {i}: c_state={c[-1].abs().mean():.4f}, hint={hint.abs().mean():.6f}")
        
        hints = list(torch.unbind(c))[:-1]
        
        if debug:
            print(f"  Final: {len(hints)} hints")
            for i, h in enumerate(hints):
                print(f"    hint[{i}]: abs_mean={h.abs().mean():.6f}")
        
        return hints


# =============================================================================
# ComfyUI Integration
# =============================================================================

class ControlNetWrapper:
    """Wrapper to integrate with ComfyUI's control system.
    
    Supports chaining multiple Flux2Fun controlnets by accumulating them
    into lists in transformer_options, which are processed by the patch.
    """
    
    def __init__(self, controlnet, control_context, strength, ctrl_h, ctrl_w):
        self.controlnet = controlnet
        self.control_context = control_context
        self.strength = strength
        self.ctrl_h = ctrl_h
        self.ctrl_w = ctrl_w
        self.previous_controlnet = None
        
        class HooksContainer:
            hooks = []
        self.extra_hooks = HooksContainer()
        
    def pre_run(self, model, percent_to_timestep_function):
        if self.previous_controlnet:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        control_prev = None
        if self.previous_controlnet:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)
        
        if transformer_options:
            # Use lists to support multiple chained Flux2Fun controlnets
            # Initialize lists if this is the first Flux2Fun controlnet in the chain
            if 'flux2_fun_controlnets' not in transformer_options:
                transformer_options['flux2_fun_controlnets'] = []
                transformer_options['flux2_fun_control_contexts'] = []
                transformer_options['flux2_fun_control_scales'] = []
                transformer_options['flux2_fun_ctrl_dims'] = []
            
            # Append this controlnet's data to the lists
            transformer_options['flux2_fun_controlnets'].append(self.controlnet)
            transformer_options['flux2_fun_control_contexts'].append(self.control_context)
            transformer_options['flux2_fun_control_scales'].append(self.strength)
            transformer_options['flux2_fun_ctrl_dims'].append((self.ctrl_h, self.ctrl_w))
        
        output = {"input": [], "output": []}
        if control_prev:
            output["input"] = control_prev.get("input", [])
            output["output"] = control_prev.get("output", [])
        return output
    
    def copy(self):
        c = ControlNetWrapper(self.controlnet, self.control_context, self.strength, self.ctrl_h, self.ctrl_w)
        c.previous_controlnet = self.previous_controlnet
        return c
    
    def cleanup(self):
        if self.previous_controlnet:
            self.previous_controlnet.cleanup()
    
    def get_models(self):
        return self.previous_controlnet.get_models() if self.previous_controlnet else []
    
    def get_extra_hooks(self):
        return self.previous_controlnet.get_extra_hooks() if self.previous_controlnet else []
    
    def inference_memory_requirements(self, dtype):
        mem = sum(p.numel() for p in self.controlnet.parameters()) * 2
        if self.previous_controlnet:
            mem += self.previous_controlnet.inference_memory_requirements(dtype)
        return mem


# =============================================================================
# ComfyUI Nodes
# =============================================================================

class Flux2FunControlNetLoader:
    """Load Flux2 Fun ControlNet checkpoint."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"controlnet_name": (folder_paths.get_filename_list("controlnet"),)}}
    
    RETURN_TYPES = ("FLUX2_FUN_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"
    
    def load_controlnet(self, controlnet_name):
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        print(f"[Flux2 Fun] Loading: {controlnet_name}")
        
        state_dict = comfy.utils.load_torch_file(controlnet_path)
        
        # Detect architecture from weights
        control_in_dim = state_dict["control_img_in.weight"].shape[1]
        hidden_size = state_dict["control_img_in.weight"].shape[0]
        num_blocks = max(int(k.split(".")[1]) for k in state_dict if k.startswith("control_transformer_blocks.")) + 1
        
        print(f"[Flux2 Fun] Architecture: hidden={hidden_size}, ctrl_dim={control_in_dim}, blocks={num_blocks}")
        
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float16
        
        controlnet = Flux2FunControlNet(
            hidden_size=hidden_size, num_attention_heads=48,
            attention_head_dim=hidden_size // 48, mlp_ratio=3.0,
            control_in_dim=control_in_dim, num_blocks=num_blocks,
            dtype=dtype, device="cpu"
        )
        
        missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"[Flux2 Fun] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Flux2 Fun] Unexpected keys: {len(unexpected)}")
        
        controlnet.to(device=device, dtype=dtype)
        controlnet.eval()
        
        print(f"[Flux2 Fun] Loaded successfully")
        return (controlnet,)


class Flux2FunControlNetApply:
    """
    Apply Flux2 Fun ControlNet to conditioning.
    
    Modes:
    - Control only: Provide control_image (pose/canny/depth/etc)
    - Control + Inpaint: Provide control_image + mask + inpaint_image
    
    Note: Inpaint-only mode (mask + inpaint without control) is supported
    but produces limited results. Use a dedicated inpaint model for best results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "controlnet": ("FLUX2_FUN_CONTROLNET",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "control_image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "conditioning/controlnet"
    
    @staticmethod
    def _patchify(x):
        """Convert [B, C, H, W] -> [B, C*4, H/2, W/2] by rearranging 2x2 patches."""
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c * 4, h // 2, w // 2)
        return x
    
    def apply_controlnet(self, conditioning, controlnet, vae, strength, 
                         control_image=None, mask=None, inpaint_image=None):
        device = comfy.model_management.get_torch_device()
        dtype = next(controlnet.parameters()).dtype
        
        # Determine dimensions from available image
        if control_image is not None:
            bs, h, w, _ = control_image.shape
        elif inpaint_image is not None:
            bs, h, w, _ = inpaint_image.shape
        else:
            raise ValueError("Must provide either control_image or inpaint_image")
        
        # Ensure dimensions divisible by 16
        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        if h != new_h or w != new_w:
            h, w = new_h, new_w
        
        # Latent dimensions (VAE outputs packed 128ch at h/16, w/16)
        lat_h, lat_w = h // 16, w // 16
        
        comfy.model_management.load_model_gpu(vae.patcher)
        
        # Process mask (ComfyUI: 1.0 = area to inpaint)
        if mask is not None:
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            
            # Binarize mask (>= 0.5 -> 1.0)
            mask_binary = (mask >= 0.5).float()
            mask_for_img = F.interpolate(mask_binary.to(device=device, dtype=dtype), (h, w), mode='nearest')
        else:
            mask_for_img = None
            mask_binary = None
        
        # Encode inpaint image (with masked region zeroed)
        if inpaint_image is not None:
            inp_img = inpaint_image[:,:,:,:3].to(device)
            if inp_img.shape[1:3] != (h, w):
                inp_img = F.interpolate(inp_img.permute(0,3,1,2), (h, w), mode='bilinear').permute(0,2,3,1)
            
            # Zero out inpaint region before encoding
            if mask_for_img is not None:
                keep_mask = (mask_for_img < 0.5).float()
                inp_img = inp_img * keep_mask.permute(0, 2, 3, 1)
            
            with torch.no_grad():
                inpaint_latents = vae.encode(inp_img)
            inpaint_flat = inpaint_latents.to(device=device, dtype=dtype).flatten(2).permute(0, 2, 1)
        else:
            inpaint_flat = torch.zeros((bs, lat_h * lat_w, 128), device=device, dtype=dtype)
        
        # Encode control image
        if control_image is not None:
            ctrl_img = control_image[:,:,:,:3].to(device)
            if ctrl_img.shape[1:3] != (h, w):
                ctrl_img = F.interpolate(ctrl_img.permute(0,3,1,2), (h, w), mode='bilinear').permute(0,2,3,1)
            with torch.no_grad():
                control_latents = vae.encode(ctrl_img)
            control_flat = control_latents.to(device=device, dtype=dtype).flatten(2).permute(0, 2, 1)
        else:
            control_flat = torch.zeros((bs, lat_h * lat_w, 128), device=device, dtype=dtype)
        
        # Process mask for control context
        if mask_binary is not None:
            mask_unpacked_size = (lat_h * 2, lat_w * 2)
            mask_for_context = F.interpolate(mask_binary.to(device=device, dtype=dtype), mask_unpacked_size, mode='nearest')
            mask_for_context = 1.0 - mask_for_context  # Invert
            mask_for_context = self._patchify(mask_for_context)
            mask_flat = mask_for_context.flatten(2).permute(0, 2, 1)
        else:
            mask_flat = torch.zeros((bs, lat_h * lat_w, 4), device=device, dtype=dtype)
        
        # Build control context: [control(128), mask(4), inpaint(128)] = 260
        control_context = torch.cat([control_flat, mask_flat, inpaint_flat], dim=2)
        
        # Determine mode for logging
        if control_image is not None and inpaint_image is not None:
            mode = "control+inpaint"
        elif control_image is not None:
            mode = "control"
        else:
            mode = "inpaint"
        
        print(f"[Flux2 Fun] Mode: {mode}, strength: {strength}")
        
        wrapper = ControlNetWrapper(controlnet, control_context, strength, lat_h, lat_w)
        
        c = [[t[0], t[1].copy()] for t in conditioning]
        for t in c:
            # Chain with existing control if present (supports multiple Flux2Fun controlnets)
            existing_control = t[1].get('control', None)
            if existing_control is not None:
                wrapper.previous_controlnet = existing_control
            t[1]['control'] = wrapper
            t[1]['control_apply_to_uncond'] = True
        
        return (c,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "Flux2FunControlNetLoader": Flux2FunControlNetLoader,
    "Flux2FunControlNetApply": Flux2FunControlNetApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2FunControlNetLoader": "Load Flux2 Fun ControlNet",
    "Flux2FunControlNetApply": "Apply Flux2 Fun ControlNet",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'Flux2FunControlNet']

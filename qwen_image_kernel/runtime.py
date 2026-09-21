# 12 GB-friendly runtime for Qwen-Image 2.1 with kernel + FP8 acceleration.
#
# VRAM orchestration (RTX 5070 12 GB):
#   phase 1: text encoder (FP8) on GPU  -> encode prompt(s) -> unload
#   phase 2: DiT (FP8, kernel blocks) on GPU -> 40 denoising steps
#   phase 3: DiT off, VAE (bf16) on GPU -> decode -> unload
#
# FP8 weights are loaded from a one-time cache (built by quantize_offline.py)
# into a meta-device skeleton, so no BF16 weight ever has to exist twice.

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import init_empty_weights
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .fp8 import FP8Linear
from .kv_cache import ContiguousKVCache
from .processors import QwenImage21KernelBlock, QwenImage21KernelProcessor

import PIL.Image as _PILImage

SYS_PROMPT = "Comprehend and analyze the provided prompt."


def calculate_shift(
    image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096,
    base_shift: float = 0.5, max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calculate_dimensions(target_area: float, ratio: float) -> tuple[int, int]:
    """(width, height) for a pixel target area preserving the given w/h ratio,
    rounded to multiples of 32 (mirrors the stock pipeline helper)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def retrieve_latents(encoder_output, generator: torch.Generator | None = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(scheduler, num_inference_steps: int, device, sigmas=None, mu: float | None = None):
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    return scheduler.timesteps, len(scheduler.timesteps)


def _fp8_shell_transformer(model: QwenImage21Transformer2DModel, kernel_blocks: bool = True) -> None:
    """FP8Linear shells for every linear; optionally swap in fused kernel blocks
    (with the exact block-causal Triton attention). With kernel_blocks=False the
    stock blocks/processor are kept - the 'FP8 only' ablation config."""
    if kernel_blocks:
        blocks = []
        for blk in model.transformer_blocks:
            kb = QwenImage21KernelBlock(blk)
            kb.attn.set_processor(QwenImage21KernelProcessor())
            blocks.append(kb)
        model.transformer_blocks = torch.nn.ModuleList(blocks)
    # remaining linears outside the blocks (img_in, txt_in, modulation, proj_out,
    # timestep embedder, norm_out.linear) get FP8 shells via the generic walk;
    # FP8Linear is not an nn.Linear subclass, so already-shelled modules pass through
    _fp8_shell_module(model)


def _fp8_shell_module(model: torch.nn.Module, skip: tuple[str, ...] = ()) -> list[str]:
    """Meta-safe FP8Linear replacement for an arbitrary module tree."""
    replaced = []
    for parent_name, parent_mod in list(model.named_modules()):
        for child_name, child in list(parent_mod.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(p in full for p in skip):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent_mod, child_name,
                        FP8Linear(child.in_features, child.out_features, bias=child.bias is not None))
                replaced.append(full)
    return replaced


@dataclass
class KernelRuntimeConfig:
    model_id: str = "Qwen/Qwen-Image-2.1"
    fp8: bool = True
    kernel_blocks: bool = True
    fp8_cache_dir: str = "fp8_cache"
    device: str = "cuda"
    verbose: bool = False
    ref_resolution: int | None = None  # side length budget for condition images; None = auto by count
    kv_v_fp8: bool = True  # store prefix-KV V in fp8e4m3 (per-head scales); K stays bf16

    def mem(self, tag: str):
        if self.verbose and torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 2**30
            print(f"[mem] {tag:36s} alloc {a:6.2f} GiB", flush=True)


class KernelRuntime:
    """Owns all components and stages them through VRAM phase by phase."""

    def __init__(self, cfg: KernelRuntimeConfig | None = None):
        self.cfg = cfg or KernelRuntimeConfig()
        self.device = torch.device(self.cfg.device)
        self.scheduler = None
        self.vae = None
        self.processor = None
        self.text_encoder = None
        self.transformer = None
        self.image_processor = None
        self._embed_cache: dict = {}

    def _fp8_cache_path(self, component: str) -> str:
        return os.path.join(self.cfg.fp8_cache_dir, f"{component}_fp8.safetensors")

    # ------------------------------------------------------------------
    # builders
    # ------------------------------------------------------------------
    def _build_transformer(self) -> QwenImage21Transformer2DModel:
        cfg = self.cfg
        config = QwenImage21Transformer2DModel.load_config(cfg.model_id, subfolder="transformer")
        params = inspect.signature(QwenImage21Transformer2DModel.__init__).parameters
        init_kwargs = {k: v for k, v in config.items() if k in params and k != "self"}

        if cfg.fp8:
            cache = self._fp8_cache_path("transformer")
            if not os.path.exists(cache):
                from .quantize_offline import build_transformer_fp8_cache
                build_transformer_fp8_cache(cfg.model_id, cache)

            # params on meta, buffers (RoPE tables etc.) created live
            with init_empty_weights(include_buffers=False):
                model = QwenImage21Transformer2DModel(**init_kwargs)
            _fp8_shell_transformer(model, kernel_blocks=cfg.kernel_blocks)

            from safetensors.torch import load_file
            sd = load_file(cache)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            assert not unexpected, f"unexpected fp8 cache keys: {unexpected[:4]}"
            assert not missing, f"missing keys after fp8 load: {missing[:8]}"
            meta_left = [n for n, t in model.named_parameters() if t.is_meta]
            assert not meta_left, f"meta params left: {meta_left[:8]}"
            return model.to(self.device)

        model = QwenImage21Transformer2DModel.from_pretrained(
            cfg.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        if cfg.kernel_blocks:
            from .processors import convert_transformer_to_kernels
            convert_transformer_to_kernels(model, use_fp8=False)
        return model.to(self.device)

    def _build_text_encoder(self):
        cfg = self.cfg
        if cfg.fp8:
            cache = self._fp8_cache_path("text_encoder")
            if not os.path.exists(cache):
                from .quantize_offline import build_text_encoder_fp8_cache
                build_text_encoder_fp8_cache(cfg.model_id, cache)

            tconfig = AutoConfig.from_pretrained(cfg.model_id, subfolder="text_encoder")
            with init_empty_weights(include_buffers=False):
                te = Qwen3VLForConditionalGeneration._from_config(tconfig)
            _fp8_shell_module(te)

            from safetensors.torch import load_file
            sd = load_file(cache)
            missing, unexpected = te.load_state_dict(sd, strict=False, assign=True)
            assert not unexpected, f"unexpected TE fp8 keys: {unexpected[:4]}"
            meta_left = [n for n, t in te.named_parameters() if t.is_meta]
            assert not meta_left, f"meta params left: {meta_left[:8]}"
            return te.to(self.device)

        te = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        if cfg.fp8:
            from .fp8 import convert_linears_fp8
            convert_linears_fp8(te)
        return te.to(self.device)

    def _ensure_static(self):
        if self.scheduler is None:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.cfg.model_id, subfolder="scheduler"
            )
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.cfg.model_id, subfolder="processor")
        if self.image_processor is None:
            self.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
        if not hasattr(self, "_dit_num_layers"):
            dcfg = QwenImage21Transformer2DModel.load_config(self.cfg.model_id, subfolder="transformer")
            self._dit_num_layers = int(dcfg["num_layers"])
            self._dit_num_heads = int(dcfg["num_attention_heads"])
            self._dit_head_dim = int(dcfg["attention_head_dim"])

    # ------------------------------------------------------------------
    # phase 1: text (+ vision) encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _image_key(image) -> str:
        import hashlib
        import io

        if isinstance(image, str):
            with open(image, "rb") as f:
                return hashlib.sha1(f.read()).hexdigest()[:16]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha1(buf.getvalue()).hexdigest()[:16]

    @torch.no_grad()
    def encode_prompt(self, prompt: str, image: list | None = None, use_cache: bool = True):
        """Encode prompt (+ condition images) with the FP8 Qwen3-VL.

        Returns (embeds [B, S, 4096] bf16, mask [B, S] bool or None,
        img_slot_mask [B, S] bool - True at the VLM's image slots).
        """
        cache_key = None
        if use_cache:
            img_key = tuple(self._image_key(i) for i in image) if image else ()
            cache_key = (prompt, img_key)
            if cache_key in self._embed_cache:
                return self._embed_cache[cache_key]

        self._ensure_static()
        if self.text_encoder is None:
            self.text_encoder = self._build_text_encoder()
        else:
            self.text_encoder.to(self.device)

        prompts = [p if p else " " for p in ([prompt] if isinstance(prompt, str) else prompt)]
        n_imgs = len(image) if image else 0

        if n_imgs == 0:
            template = (
                f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{{}}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            filled = [template.format(p) for p in prompts]
            condition_pil_list = None
        else:
            # vision placeholders: one per condition image, in order
            replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(2, n_imgs + 1):
                replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
            ti2i = (
                f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{replace}{{}}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            filled = [ti2i.format(p) for p in prompts]
            # the checkpoint was trained with RGBA flattened over white for the
            # vision encoder; the VAE still reads all four channels
            condition_pil_list = []
            for img in image:
                if isinstance(img, str):
                    img = _PILImage.open(img)
                elif not isinstance(img, _PILImage.Image):
                    img = _PILImage.fromarray(img)
                if img.mode == "RGBA":
                    white = _PILImage.new("RGB", img.size, (255, 255, 255))
                    white.paste(img, mask=img.getchannel("A"))
                    img = white
                condition_pil_list.append(img)

        processor_kwargs = {
            "text": filled,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if condition_pil_list is not None:
            processor_kwargs["images"] = condition_pil_list

        model_inputs = self.processor(**processor_kwargs).to(self.device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
            "logits_to_keep": 1,
        }
        if condition_pil_list is not None and hasattr(model_inputs, "pixel_values"):
            forward_kwargs["pixel_values"] = model_inputs.pixel_values
            forward_kwargs["image_grid_thw"] = model_inputs.image_grid_thw
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        te = self.text_encoder
        text_model = getattr(te.model, "language_model", te.model)
        handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            outputs = te(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = outputs.hidden_states[-1]

        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYS_PROMPT}]}]
        sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        drop = len(sys_tokens[0])

        bool_mask = model_inputs.attention_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        splits = [e[drop:] for e in torch.split(selected, valid_lengths.tolist(), dim=0)]

        img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        n = len(prompts)
        img_pad = [
            (model_inputs.input_ids[i][bool_mask[i]] == img_token_id)[drop:] for i in range(n)
        ]

        max_len = max(e.shape[0] for e in splits)
        embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.shape[0], u.shape[1])]) for u in splits
        ])
        mask = torch.stack([
            torch.cat([torch.ones(u.shape[0], dtype=torch.bool, device=u.device),
                       torch.zeros(max_len - u.shape[0], dtype=torch.bool, device=u.device)])
            for u in splits
        ])
        img_pad = torch.stack([
            torch.cat([u, u.new_zeros(max_len - u.shape[0], dtype=torch.bool)]) for u in img_pad
        ])
        if mask.all():
            mask = None

        if use_cache and cache_key is not None:
            self._embed_cache[cache_key] = (embeds, mask, img_pad)

        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        return embeds, mask, img_pad

    # ------------------------------------------------------------------
    # phase 2 + 3: denoise + decode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image: list | None = None,
        height: int | None = None,
        width: int | None = None,
        output_resolution: int = 1024,
        num_inference_steps: int = 40,
        seed: int | None = None,
        true_cfg_scale: float = 1.0,
        negative_prompt: str | None = None,
        output_type: str = "pil",
        step_callback=None,
    ):
        """Text-to-image (image=None) or image-conditioned generation.

        `image`: one or more condition images (PIL / numpy / path). The output
        size defaults to the last condition image's aspect ratio at
        `output_resolution^2` pixels.

        Reference images are resized so their area matches a per-image budget
        that keeps the prefix KV cache within 12 GB VRAM: 1024^2 for one
        image, 768^2 for two, 512^2 for three or more (override with
        `ref_resolution`).
        """
        self._ensure_static()
        # ---- normalize condition images --------------------------------
        condition_images: list = []
        if image is not None:
            image = image if isinstance(image, list) else [image]
            for img in image:
                if isinstance(img, str):
                    img = _PILImage.open(img)
                elif isinstance(img, np.ndarray):
                    img = _PILImage.fromarray(img)
                if not isinstance(img, _PILImage.Image):
                    raise ValueError(f"unsupported condition image type {type(img)!r}")
                condition_images.append(img)

        if condition_images:
            last_w, last_h = condition_images[-1].size
            if height is None or width is None:
                w, h = calculate_dimensions(output_resolution * output_resolution, last_w / last_h)
                width = width or w
                height = height or h
        height = (height or output_resolution) // 32 * 32
        width = (width or output_resolution) // 32 * 32

        n_refs = len(condition_images)
        if condition_images:
            ref_area = (self.cfg.ref_resolution or output_resolution) ** 2
            if self.cfg.ref_resolution is None:
                # auto: keep the prefix KV cache and working set inside the
                # 12 GB budget (each 16px latent token costs ~512 KB of
                # bf16 prefix cache across 32 layers)
                ref_area = {1: 768, 2: 640}.get(min(n_refs, 3), 512) ** 2
            input_image_sizes = []
            input_images, vae_images = [], []
            for img in condition_images:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                w, h = calculate_dimensions(ref_area, img.size[0] / img.size[1])
                input_image_sizes.append((w, h))
                input_images.append(self.image_processor.resize(img, width=w, height=h))
                vae_images.append(
                    self.image_processor.preprocess(img, width=w, height=h).unsqueeze(2)
                )

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # --- phase 1: all prompt encoding happens before the DiT loads ---
        embeds, embeds_mask, img_pad_mask = self.encode_prompt(prompt, image=input_images if condition_images else None)
        do_cfg = true_cfg_scale > 1 and negative_prompt is not None
        neg_embeds = neg_mask = neg_img_pad = None
        if do_cfg:
            neg_embeds, neg_mask, neg_img_pad = self.encode_prompt(negative_prompt, image=input_images if condition_images else None)
        batch = embeds.shape[0]

        # --- phase 1b: VAE-encode the condition images -------------------
        # The VAE runs in fp32 (its checkpoint dtype, matching the reference
        # implementation) - bf16 encode is measurably lossier and the edit
        # path conditions directly on these latents.
        input_images_latents = None
        if condition_images:
            if self.vae is None:
                self.vae = AutoencoderKLQwenImage21.from_pretrained(
                    self.cfg.model_id, subfolder="vae", torch_dtype=torch.float32
                )
            self.vae.to(self.device, dtype=torch.float32)
            self.cfg.mem("VAE on (encode refs)")
            num_channels = self.vae.config.z_dim
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, num_channels, 1, 1, 1).to(self.device, torch.float32)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, num_channels, 1, 1, 1).to(self.device, torch.float32)
            packed = []
            for vae_img in vae_images:
                img = vae_img.to(device=self.device, dtype=torch.float32)
                encoded = retrieve_latents(self.vae.encode(img), generator=generator, sample_mode="argmax")
                encoded = (encoded - latents_mean) / latents_std
                z_h, z_w = encoded.shape[3:]
                packed.append(encoded.view(batch, num_channels, z_h * z_w).transpose(1, 2))
            input_images_latents = torch.cat(packed, dim=1).to(torch.bfloat16)
            self.vae.to("cpu")
            torch.cuda.empty_cache()
            self.cfg.mem("condition latents encoded")

        # --- phase 1c: prefix KV cache slab ------------------------------
        # One contiguous allocation BEFORE the DiT weights land in VRAM. A
        # cache scattered across a fragmented pool makes the attention kernel
        # read at a fraction of bandwidth under WDDM (observed 6.5x step
        # slowdown at 1024^2 references). The slab persists on the runtime and
        # is reused across generations (re-allocating it per generation
        # re-fragments the pool and reintroduces the slowdown).
        def _get_kv_pool(img_mask_vlm, s_vlm):
            if not self.cfg.kernel_blocks:
                return QwenImage21KVCache(32)
            n_slots = int(img_mask_vlm[:, :s_vlm].sum())
            needed = s_vlm + 3 * n_slots + 16
            pool = getattr(self, "_kv_pool", None)
            if pool is None or pool.capacity < needed or pool.batch_size != batch:
                pool = ContiguousKVCache(
                    self._dit_num_layers, batch, needed + 256,
                    self._dit_num_heads, self._dit_head_dim,
                    v_fp8=self.cfg.kv_v_fp8 and self.cfg.fp8, device=self.device,
                )
                self._kv_pool = pool
            return pool

        cond_cache = _get_kv_pool(img_pad_mask, embeds.shape[1])
        neg_cache = None
        if do_cfg:
            neg_cache = self._get_kv_pool(neg_img_pad, neg_embeds.shape[1])
            if neg_cache is cond_cache:
                # CFG needs two independent caches
                neg_cache = ContiguousKVCache(
                    self._dit_num_layers, batch, cond_cache.capacity,
                    self._dit_num_heads, self._dit_head_dim,
                    v_fp8=self.cfg.kv_v_fp8 and self.cfg.fp8, device=self.device,
                )

        # --- phase 2: DiT ---
        if self.transformer is None:
            self.transformer = self._build_transformer()
        elif next(self.transformer.parameters()).device.type != self.cfg.device:
            self.transformer.to(self.device)

        num_channels_latents = 64
        vae_scale = 16
        lat_h, lat_w = 2 * (height // (vae_scale * 2)), 2 * (width // (vae_scale * 2))
        latents = randn_tensor(
            (batch, 1, num_channels_latents, lat_h, lat_w), generator=generator, device=self.device, dtype=torch.bfloat16
        )
        latents = latents.view(batch, num_channels_latents, lat_h * lat_w).transpose(1, 2)
        seq_len = latents.shape[1]

        img_shapes = [
            [*[(1, h // vae_scale, w // vae_scale) for (w, h) in input_image_sizes],
             (1, lat_h, lat_w)]
            for _ in range(batch)
        ] if condition_images else [[(1, lat_h, lat_w)] for _ in range(batch)]
        target_slots = seq_len // 4
        image_pad_mask = torch.cat([img_pad_mask, img_pad_mask.new_ones(batch, target_slots)], dim=1)
        if do_cfg:
            neg_img_pad = torch.cat([neg_img_pad, neg_img_pad.new_ones(batch, target_slots)], dim=1)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, sigmas=sigmas, mu=mu
        )

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            kv_mode = "extract" if i == 0 else "cached"
            timestep = t.expand(batch).to(torch.bfloat16)

            latent_model_input = latents
            if input_images_latents is not None:
                latent_model_input = torch.cat([input_images_latents, latents], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=embeds_mask,
                img_shapes=img_shapes,
                img_mask=image_pad_mask,
                attention_kwargs=None,
                kv_cache=cond_cache,
                kv_cache_mode=kv_mode,
                return_dict=False,
            )[0][:, -seq_len:]

            if do_cfg:
                neg_noise = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask,
                    img_shapes=img_shapes,
                    img_mask=neg_img_pad,
                    attention_kwargs=None,
                    kv_cache=neg_cache,
                    kv_cache_mode=kv_mode,
                    return_dict=False,
                )[0][:, -seq_len:]
                noise_pred = neg_noise + true_cfg_scale * (noise_pred - neg_noise)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if step_callback is not None:
                step_callback(i, t)

        # --- phase 3: VAE decode ---
        self.transformer.to("cpu")
        del cond_cache, neg_cache, noise_pred
        torch.cuda.empty_cache()
        self.cfg.mem("DiT off, before VAE")
        if self.vae is None:
            self.vae = AutoencoderKLQwenImage21.from_pretrained(
                self.cfg.model_id, subfolder="vae", torch_dtype=torch.float32
            )
        self.vae.to(self.device, dtype=torch.float32)
        # non-tiled 1024px decode needs >20 GiB of activations; 512px tiles @ 384
        # stride is the configuration the vLLM recipe validated for this VAE
        self.vae.enable_tiling(
            tile_sample_min_height=512, tile_sample_min_width=512,
            tile_sample_stride_height=384, tile_sample_stride_width=384,
        )
        self.cfg.mem("VAE on, tiling on")

        latents = latents.transpose(1, 2).reshape(batch, num_channels_latents, 1, lat_h, lat_w)
        latents = latents.to(self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, self.vae.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, self.vae.dtype)
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.image_processor.postprocess(image, output_type=output_type)

        self.vae.to("cpu")
        torch.cuda.empty_cache()
        return images

    # backwards-compatible alias
    def generate_text_to_image(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)
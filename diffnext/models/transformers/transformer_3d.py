# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Base 3D transformer model for video generation."""

from typing import Dict
import torch
from torch import nn
from tqdm import tqdm


class Transformer3DModel(nn.Module):
    """Base 3D transformer model for video generation."""

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        video_pos_embed=None,
        image_pos_embed=None,
        motion_embed=None,
        noise_scheduler=None,
        sample_schduler=None,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder = video_encoder
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.mask_embed = mask_embed
        self.text_embed = text_embed
        self.video_pos_embed = video_pos_embed
        self.image_pos_embed = image_pos_embed
        self.motion_embed = motion_embed
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_schduler
        self.pipeline_preprocess = lambda inputs: inputs
        self.loss_repeat = 4

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    def preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        add_guidance = inputs.get("guidance_scale", 1) > 1
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        
        if inputs.get("x", None) is None:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
            
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
            
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
            
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        if video_shape is not None:
            loss = loss.view((-1,) + video_shape).transpose(0, 1).sum((1, 2))
            i2i = loss[1:].sum().mul_(video_shape[0] / (video_shape[0] - 1))
            return {"loss_t2i": loss[0].mul(video_shape[0]), "loss_i2i": i2i}
        return {"loss": loss.sum()}

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """Generate a batch of frames."""
        
        guidance_scale = inputs.get("guidance_scale", 1)
        min_guidance_scale = inputs.get("min_guidance_scale", guidance_scale) # guidance_scale
        max_guidance_scale = inputs.get("max_guidance_scale", guidance_scale) # guidance_scale
        guidance_end = max_guidance_scale if states["t"] else guidance_scale
        guidance_start = max_guidance_scale if states["t"] else min_guidance_scale
        generator = self.mask_embed.generator = inputs.get("generator", None)
        
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        forcing_set_num = None if not inputs['forcing_set'] else all_num_preds[0]
        
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None # [2, 1024, 1024], [1, 4, 128, 128], None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None # None
          
        # [1, 4, 6, 9, 11, 13, 16, 19, 21, 23, 25, 28, 31, 32, 35, 38, 39, 42, 44, 47, 48, 51, 52, 55, 57, 59, 61, 63, 64, 67, 68, 71, 72, 73, 76, 77, 78, 80, 81, 83, 85, 85, 87, 88, 89, 91, 91, 93, 93, 94, 95, 96, 97, 97, 98, 98, 99, 99, 100, 100, 100, 100, 100, 101]
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            # always guidance_scale
            guidance_level = (i + 1) / len(all_num_preds)
            guidance_scale = (guidance_end - guidance_start) * guidance_level + guidance_start
            
            z = self.mask_embed(self.image_encoder.patch_embed(x)) # [1, 4096, 1024]
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds, forcing_set_num=forcing_set_num) # [1, 4096, 1], [1, L, 1]
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids # [2, L, 1]
            z = torch.cat([z] * 2) if guidance_scale > 1 else z # [2, 4096, 1024]
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = self.image_encoder(z, c, prev_ids, pos=pos) # [2, 4096, 1024]
            
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator) # [1, 4, 128, 128]
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids) # [1, 4096, 16]
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask))) # [1, 4, 128, 128]
        return None
            
    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1) # 1
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25)) # 25
        states = {"x": inputs["x"], "noise": inputs["x"].clone()} # [1, 4, 128, 128]
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        
        states['x_cond'] = inputs['x_cond']
        
        if self.image_pos_embed: #None
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length) #[1, 1, 1024]
            
        [setattr(blk.attn, "cache_kv", max_latent_length > 1) for blk in self.video_encoder.blocks]
        
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)): # max_latent_length=1, tqdm1=False
            
            pos = time_pos[states["t"]] if time_pos else None #None
            c = self.video_encoder.patch_embed(states["x"]) # [1, 1024, 1024]
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c # [1, 1024, 1024]
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c # [1, 1024, 1024]
            c = torch.cat([c] * 2) if guidance_scale > 1 else c # [2, 1024, 1024]
            c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos) # [2, 1024, 1024]
            
            if not isinstance(self.video_encoder.mixer, torch.nn.Identity): # torch.nn.Identity
                states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
                states["*"] = states["*"] if states["t"] else states["c"]
                
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
                
        [setattr(blk.attn, "cache_kv", False) for blk in self.video_encoder.blocks]
        return None

    def train_video(self, inputs):
        """Train a batch of videos."""
        # 3D temporal autoregressive modeling (TAM).
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1])
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1)
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None
        pos = self.video_pos_embed.get_pos(latent_length, bs) if self.image_pos_embed else pos
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length
        pos = self.image_pos_embed.get_pos(1, bs) if self.image_pos_embed else pos
        z = self.image_encoder(self.mask_embed(z), c.reshape(bs, -1, c.size(-1)), pos=pos)
        # 1D token-wise diffusion modeling (MLP).
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return self.get_losses(z, x, video_shape=video_shape)
    
    def forward(self, inputs):
        """Define the computation performed at every call."""
        self.pipeline_preprocess(inputs)
        self.preprocess(inputs)
        if self.training:
            return self.train_video(inputs)
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_video(inputs)
        return torch.stack(inputs["latents"], dim=2)
    
    @torch.no_grad()
    def denoise(self, z, x, guidance_scale=1, generator=None, pred_ids=None) -> torch.Tensor:
        """
        Run diffusion denoising process.

        Args:

        Returns:
        """
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            x_pack = torch.cat([x] * 2) if guidance_scale > 1 else x
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            noise_pred = self.image_decoder(x_pack, timestep, z, pred_ids) # [2, 4096, 16]
            if guidance_scale > 1:
                cond, uncond = noise_pred.chunk(2)
                noise_pred = uncond.add_(cond.sub_(uncond).mul_(guidance_scale)) # [1, 4096, 16]
            noise_pred = self.image_encoder.patch_embed.unpatchify(noise_pred) #[1, 4, 128, 128]
            x = self.sample_scheduler.step(noise_pred, t, x, generator=generator).prev_sample #[1, 4, 128, 128]
        return self.image_encoder.patch_embed.patchify(x)
            
    @torch.inference_mode()
    def generate_image_inpaint(self, inputs: Dict):
        """Generate a batch of videos."""
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1) # 1
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25)) # 25
        states = {"x": inputs["x"], "noise": inputs["x"].clone()} # [1, 4, 64, 64]
        masks = inputs["masks"].to(device=self.device) # [1, 1, 32, 32]
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        envtokennums = inputs['envtokennums']
        cond_para = inputs['cond_para']
        generator = self.mask_embed.generator = inputs.get("generator", None)
        
        prev_ids, _, _ = self.mask_previds_sample(masks, envtokennums, generator)
        print('The number of environment tokens is: ', envtokennums)
        print('The shape of prev_ids tokens is: ', prev_ids.shape)
        prev_ids = torch.cat([prev_ids] * 2) if guidance_scale > 1 else prev_ids
    
        if self.image_pos_embed: #None
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length) #[1, 1, 1024]
            
        [setattr(blk.attn, "cache_kv", max_latent_length > 1) for blk in self.video_encoder.blocks]
        
        states["t"] = 0
        
        pos = time_pos[states["t"]] if time_pos else None #None
        c = self.video_encoder.patch_embed(states["x"]) # [1, 256, 1024]
        c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
        c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c # [1, 256, 1024]
        c = torch.cat([c] * 2) if guidance_scale > 1 else c # [2, 256, 1024]
        
        c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos) # [2, 256, 1024]
        
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity): # torch.nn.Identity
            states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
            states["*"] = states["*"] if states["t"] else states["c"]
            
        guidance_scale = inputs.get("guidance_scale", 1)
        
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        
        c, x, self.mask_embed.mask = states["c"], states["x"], None # [2, 256, 1024], [1, 4, 64, 64], None
        self.image_encoder.patch_embed.height = x.size(-2) // self.image_encoder.patch_embed.patch_size
        self.image_encoder.patch_embed.width = x.size(-1) // self.image_encoder.patch_embed.patch_size
        x = self.image_encoder.patch_embed.patchify(x)
        masks = masks.reshape(-1, 1, masks.shape[-2]*masks.shape[-1]).permute(0, 2, 1)
        x = x * (1 - masks)
        x = self.image_encoder.patch_embed.unpatchify(x)
        
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None # None
        
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            
            z = self.mask_embed.inpaint(self.image_encoder.patch_embed(x), masks) # [1, 1024, 1024]
            
            pred_mask, pred_ids, inpaint_token_ids, uninpaint_token_ids = self.mask_embed.get_inpaint_pred_mask(num_preds) # [1, 1024, 1], [1, 1, 1]
            inpaint_token_ids = inpaint_token_ids if i != 0 else uninpaint_token_ids.clone()
            inpaint_token_ids, uninpaint_token_ids = inpaint_token_ids + c.size(1), uninpaint_token_ids + c.size(1)
            
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids # [2, 1, 1]
            z = torch.cat([z] * 2) if guidance_scale > 1 else z # [2, 1024, 1024]
            prev_ids = prev_ids if i else prev_ids.to(device=pred_ids.device, dtype=pred_ids.dtype)
            z = self.image_encoder.inpaint_forward(
                z, c, prev_ids, cond_para=cond_para, pos=pos,
                inpaint_token_ids=inpaint_token_ids, uninpaint_token_ids=uninpaint_token_ids,
                atten_entro_value=inputs['atten_entro_value'],
            )
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator) # [1, 4, 64, 64]
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids) # [1, 1024, 16]
            
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask))) # [1, 4, 64, 64]

        latents.append(x)
                
        [setattr(blk.attn, "cache_kv", False) for blk in self.video_encoder.blocks]
        return None
        
    def mask_previds_sample(self, masks, num_tokens, generator, num_tokens_max=1024, dilate_iter=15):
        """Sample a batch of masked tokens."""
        mask_background = torch.ones_like(masks) - masks
        mask_background = mask_background.reshape(-1, 1, mask_background.shape[-2]*mask_background.shape[-1]).permute(0, 2, 1)
            
        background_indices_list = []
        for batch in mask_background:
            indices = (batch.squeeze() == 1).nonzero(as_tuple=True)[0].unsqueeze(-1)
            background_indices_list.append(indices)
        all_background_ids = torch.stack(background_indices_list, dim=0)
    
        if num_tokens == 0:
            print('|Warning| You only use the text prompt tokens as input for encoder.')
            return torch.empty((masks.size(0), 0, 1)), torch.zeros_like(masks), all_background_ids.to(torch.int64)
        
        if num_tokens == num_tokens_max:
            print('|Warning| You use all background tokens and the text prompt tokens as input for encoder.')
            prev_masks = mask_background.permute(0, 2, 1).reshape(
                -1, 1, masks.shape[-2], masks.shape[-1]
            )
            return all_background_ids.to(torch.int64), prev_masks, all_background_ids.to(torch.int64)

        mask_dilated = masks.clone()
        max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        for i in range(200):
            mask_dilated = max_pool(mask_dilated)
            mask_dilated_temp = torch.sum((mask_dilated - masks), dim=(-1, -2, -3))
            if mask_dilated_temp[0].item() > num_tokens:
                break
        mask_dilated = mask_dilated - masks
        mask_dilated = mask_dilated.reshape(-1, 1, mask_dilated.shape[-2]*mask_dilated.shape[-1]).permute(0, 2, 1)
        
        zero_indices_list = []
        for batch in mask_dilated:
            indices = (batch.squeeze() == 1).nonzero(as_tuple=True)[0].unsqueeze(-1)
            zero_indices_list.append(indices)
        prev_ids = torch.stack(zero_indices_list, dim=0)
        prev_ids = prev_ids[:, :num_tokens, :]
            
        return prev_ids.to(torch.int64), mask_dilated.permute(0, 2, 1).reshape(-1, 1, masks.shape[-2], masks.shape[-1]), all_background_ids.to(torch.int64)
        
    def inpaint_preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        add_guidance = inputs.get("guidance_scale", 1) > 1
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        
        if inputs.get("x", None) is None:
            raise ValueError("Missing input tensor: x.")
        else:
            assert inputs["x"].shape[0] == inputs["masks"].shape[0]
            assert all(torch.sum(inputs["masks"], dim=(-1, -2)) > 0)
        
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
            
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
            
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]
    
    def inpaint(self, inputs):
        """Define the computation performed at every call."""
        self.pipeline_preprocess(inputs)
        self.inpaint_preprocess(inputs)
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_image_inpaint(inputs)
        return torch.stack(inputs["latents"], dim=2)
    
    

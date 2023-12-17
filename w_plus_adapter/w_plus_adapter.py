import os
from typing import List

import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

from .utils import is_torch2_available

if is_torch2_available():
    from .attention_processor import WPlusAttnProcessor2_0 as WPlusAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import WPlusAttnProcessor, AttnProcessor



class WProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=768, embeddings_dim=512, clip_extra_context_tokens=4):
        super().__init__()
        self.proj_index = [4, 8, 12, 18]
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj_0 = torch.nn.Linear(embeddings_dim*self.proj_index[0],  cross_attention_dim, bias=False) #
        self.proj_1 = torch.nn.Linear(embeddings_dim*(self.proj_index[1] - self.proj_index[0]),  cross_attention_dim, bias=False)
        self.proj_2 = torch.nn.Linear(embeddings_dim*(self.proj_index[2] - self.proj_index[1]),  cross_attention_dim, bias=False)
        self.proj_3 = torch.nn.Linear(embeddings_dim*(self.proj_index[3] - self.proj_index[2]),  cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, w_embeds):
        embeds = w_embeds 
        B, _, _ = embeds.size()
        clip_w_plus_0_tokens = self.proj_0(embeds[:, :self.proj_index[0], :].view(B, -1)).unsqueeze(1)
        clip_w_plus_1_tokens = self.proj_1(embeds[:, self.proj_index[0]:self.proj_index[1], :].view(B, -1)).unsqueeze(1)
        clip_w_plus_2_tokens = self.proj_2(embeds[:, self.proj_index[1]:self.proj_index[2], :].view(B, -1)).unsqueeze(1)
        clip_w_plus_3_tokens = self.proj_3(embeds[:, self.proj_index[2]:self.proj_index[3], :].view(B, -1)).unsqueeze(1)
        clip_extra_context_tokens = torch.cat([clip_w_plus_0_tokens, clip_w_plus_1_tokens, clip_w_plus_2_tokens, clip_w_plus_3_tokens], dim=1)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class WPlusAdapter:
    
    def __init__(self, sd_pipe, wplus_ckpt, device):
        
        self.device = device
        self.wplus_ckpt = wplus_ckpt
        self.num_tokens = 4
        
        self.pipe = sd_pipe.to(self.device)
        self.set_wplus_adapter()
        
        self.clip_image_processor = CLIPImageProcessor()
        # w proj model
        self.w_proj_model = self.init_proj()
        
        self.load_wplus_adapter()
        
    def init_proj(self):
        w_proj_model = WProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            embeddings_dim=512,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return w_proj_model
        
    def set_wplus_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
                
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = WPlusAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=1.0,num_tokens= self.num_tokens).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        
    def load_wplus_adapter(self):
        state_dict = torch.load(self.wplus_ckpt, map_location="cpu")
        self.w_proj_model.load_state_dict(state_dict["w_proj"])
        wplus_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        wplus_layers.load_state_dict(state_dict["wplus_adapter"])


    
    @torch.inference_mode()
    def get_w_embeds(self, w):
        w_prompt_embeds = self.w_proj_model(w)
        uncond_w_prompt_embeds = self.w_proj_model(torch.zeros_like(w))
        return w_prompt_embeds, uncond_w_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, WPlusAttnProcessor):
                attn_processor.scale = scale
        

    def generate_idnoise(
        self,
        prompt='a photo of face',
        negative_prompt=None,
        w=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        guidance_scale=7.5,
        num_inference_steps=30,
        use_freeu=False,
        **kwargs,
    ):
        self.set_scale(scale)
        
        num_prompts = num_samples
        
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        w_prompt_embeds, uncond_w_prompt_embeds = self.get_w_embeds(w)
        bs_embed, seq_len, _ = w_prompt_embeds.shape
        w_prompt_embeds = w_prompt_embeds.repeat(1, num_samples, 1)
        w_prompt_embeds = w_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_w_prompt_embeds = uncond_w_prompt_embeds.repeat(1, num_samples, 1)
        uncond_w_prompt_embeds = uncond_w_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
  
            prompt_embeds = torch.cat([prompt_embeds_, w_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_w_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        if use_freeu:
            self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        else:
            self.pipe.disable_freeu()
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
     
        
        return images
    


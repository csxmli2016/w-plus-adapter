import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import csv

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from w_plus_adapter.w_plus_adapter import WProjModel
from w_plus_adapter.utils import is_torch2_available

if is_torch2_available():
    from w_plus_adapter.attention_processor import WPlusAttnProcessor2_0 as WPlusAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from w_plus_adapter.attention_processor import WPlusAttnProcessor, AttnProcessor

import os.path as osp
import numpy as np
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file=None, tokenizer=None, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path


        self.noise = []
        self.image = []
        self.wild = []
        self.caption = []
        self.wild_mask = []

        imgs_path = './Face/FFHQ512'
        noise_path = './Face/FFHQ512_e4e_nobg_w_plus'
        wilds_path = './Face/FFHQ-in-the-wild/in-the-wild-min-512'
        wilds_face_path = './Face/FFHQ-in-the-wild/in-the-wild-min-512-face-mask'
        captions_path = './Face/FFHQ-in-the-wild/captions_blip2'

        f = open('./Face/ffhq_wild_names_with_caption.txt', 'r')
        lines = f.readlines()
        for l in lines:
            if len(l) > 0:
                img_name = l.strip()
                self.image.append(osp.join(imgs_path, img_name))
                self.noise.append(osp.join(noise_path, img_name[:-4]+'.pth'))
                self.wild.append(osp.join(wilds_path, img_name))
                self.caption.append(osp.join(captions_path, img_name[:-4]+'.txt'))
                self.wild_mask.append(osp.join(wilds_face_path, img_name))



        self.noise_shhq = []
        self.image_shhq = []
        self.wild_shhq = []
        self.caption_shhq = []
        self.wild_mask_shhq = []

        imgs_path_shhq = './Face/SHHQ/face_images' 
        noise_path_shhq = './Face/SHHQ/face_e4e_nobg_w_plus'
        masks_path_shhq = './Face/SHHQ/face_mask'
        wilds_path_shhq = './Face/SHHQ/wild_images_2048_1024'
        wilds_face_path_shhq = './Face/SHHQ/wild_images_face_region_mask_2048_1024'
        captions_path_shhq = './Face/SHHQ/wild_captions'

        f = open('./Face/shhq_wild_caption.txt', 'r')
        lines = f.readlines()
        for l in lines:
            if len(l) > 0:
                img_name = l.strip()
                self.image_shhq.append(osp.join(imgs_path_shhq, img_name))
                self.noise_shhq.append(osp.join(noise_path_shhq, img_name[:-4]+'.pth'))
                self.wild_shhq.append(osp.join(wilds_path_shhq, img_name))
                self.caption_shhq.append(osp.join(captions_path_shhq, img_name[:-4]+'.txt'))
                self.wild_mask_shhq.append(osp.join(wilds_face_path_shhq, img_name))

        
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.transform_wild = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),

        ])
        self.clip_image_processor = CLIPImageProcessor()
        self.data = self.image

    
    def aug_self(self, img=None, mask=None, dataset_type=2):
        '''
        2 is shhq
        0 is ffhq
        '''
        img = np.array(img) #PIL, rgb, 0~255
        mask = np.array(mask)
        h, w = img.shape[:2]

        if dataset_type == 2: #SHHQ 1024*2048
            center = (w//2,h//12)
            scale = random.randint(10, 16)/10.0

        else:# ffhq #min 512
            center = (w//2,h//4)
            scale = random.randint(8, 12)/10.0

        angle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=random.choice([cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101]))
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0])
        
        select_scale = random.randint(512, 560) / min(h, w) 
        img = cv2.resize(img, (int(w * select_scale), int(h * select_scale)), cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (int(w * select_scale), int(h * select_scale)), cv2.INTER_NEAREST)

        nonzero_indices = np.argwhere(mask == 255)

        h0, w0 = img.shape[:2]
        try:
            min_x = max(np.min(nonzero_indices[:, 1]), 1)
            min_y = min(max(np.min(nonzero_indices[:, 0]), 1), h0//12)
            w_offset = np.random.choice(np.arange(0, min(min_x, w0-512+1)))
            h_offset = np.random.choice(np.arange(0, min(min_y, h0-512+1)))
        except:
            h_offset = np.random.choice(np.arange(0, h0-512+1))
            w_offset = np.random.choice(np.arange(0, w0-512+1))

        img = img[h_offset:h_offset+512, w_offset:w_offset+512, :]
        mask = mask[h_offset:h_offset+512, w_offset:w_offset+512]

        area_face = mask.mean()

        image = Image.fromarray(img)    
        mask = Image.fromarray(mask)
        return image, mask, area_face


    def __getitem__(self, idx):

        which_data = random.random()
        dataset_type = 2
        if which_data > 0.55: #FFHQ
            ind = random.randint(0, len(self.image) - 1)
            # ind = 37354
            image_file = self.image[ind]
            noise_file = self.noise[ind]
            wild_file = self.wild[ind]
            caption_file = self.caption[ind]
            wild_mask_file = self.wild_mask[ind]
            dataset_type = 0
        elif which_data > 0: #SHHQ
            ind = random.randint(0, len(self.image_shhq) - 1)
            # ind = 11195
            image_file = self.image_shhq[ind]
            noise_file = self.noise_shhq[ind]
            wild_file = self.wild_shhq[ind]
            caption_file = self.caption_shhq[ind]
            wild_mask_file = self.wild_mask_shhq[ind]
            dataset_type = 2
        
        # read noise
        noise = torch.load(noise_file, map_location=torch.device('cpu'))
        noise.requires_grad = False
        # read image
        raw_image = Image.open(image_file).convert("RGB")
        image = self.transform(raw_image)

        wild_image = Image.open(wild_file).convert("RGB")
        wild_mask = Image.open(wild_mask_file)
        
        wild_image, wild_mask, face_area_mean = self.aug_self(wild_image, wild_mask, dataset_type)

        #'erode and blur'
        
        if face_area_mean > 200:
            erode_size = random.randint(150, 180)
            kernel_size = random.randint(60, 80)
        elif face_area_mean > 120:
            erode_size = random.randint(110, 160)
            kernel_size = random.randint(60, 80)
        elif face_area_mean > 80:
            erode_size = random.randint(100, 120)
            kernel_size = random.randint(40, 60)
        elif face_area_mean > 50:
            erode_size = random.randint(80, 110)
            kernel_size = random.randint(30, 50)
        elif face_area_mean > 30:
            erode_size = random.randint(40, 80)
            kernel_size = random.randint(20, 40)
        elif face_area_mean > 20:
            erode_size = random.randint(20, 50)
            kernel_size = random.randint(10, 20)
        else:
            erode_size = random.randint(10, 30)
            kernel_size = random.randint(10, 20)

        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_2 = np.ones((erode_size, erode_size), dtype=np.uint8) # 
        wild_mask = cv2.erode(np.array(wild_mask), kernel_2, 1) 
        
        wild_mask = cv2.GaussianBlur(wild_mask, (kernel_size, kernel_size), 0)
        wild_mask = Image.fromarray(wild_mask.astype(np.uint8)) 


        wild_image = self.transform_wild(wild_image)
        wild_mask = self.transform_mask(wild_mask)

        captions = []
        with open(caption_file, 'r') as f:
            cps = f.readlines()
            for cp in cps:
                cp = cp.strip()
                if random.random() > 0.1:
                    cp_fix = cp
                else:
                    cp_fix = cp
                captions.append(cp_fix)
        
        if len(captions) == 0:
            caption = 'a person'
            print(image_file)
        else:
            caption = captions[random.randint(0, len(captions)-1)]
  
        text = caption

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        _, id_name = osp.split(image_file)
        
        return {
            "image": image,
            "wild_image": wild_image,
            "wild_mask": wild_mask,
            "text_input_ids": text_input_ids,
            "text": text+str(dataset_type)+'_'+str(int(face_area_mean))+'_'+str(erode_size)+'_'+str(kernel_size),
            "drop_image_embed": drop_image_embed,
            "id_noise": noise,
            'id_name': id_name[:-4]
        }

    def __len__(self):
        return len(self.image_shhq) + len(self.image)
    
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    wild_images = torch.stack([example["wild_image"] for example in data])
    wild_masks = torch.stack([example["wild_mask"] for example in data])
    idnoises = torch.stack([example["id_noise"] for example in data])
    idnames = [example["id_name"] for example in data]
    text = [example["text"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "wild_images": wild_images,
        "wild_masks": wild_masks,
        "text_input_ids": text_input_ids,
        "drop_image_embeds": drop_image_embeds,
        "id_noises": idnoises,
        'id_names': idnames,
        'text': text,
    }


class WPlusAdapter(torch.nn.Module):
    """WPlus-Adapter"""
    def __init__(self, unet, w_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.w_proj_model = w_proj_model
        self.adapter_modules = adapter_modules

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, WPlusAttnProcessor):
                attn_processor.scale = scale

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        wplus_tokens = self.w_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, wplus_tokens], dim=1)
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-wplus_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="tb_logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--learning_rate_proj",
        type=float,
        default=2e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=600)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--lambda_disen", type=float, default=0.6, help="loss weight.")
    parser.add_argument("--lambda_preserve", type=float, default=0.4, help="loss weight.")
    parser.add_argument("--stage1_pretrain", type=str, default=None, help="pre-trained stage1 models.")
    
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    


def tensor2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

def tensor2mask(mask):
    mask = mask.clamp(0, 1)
    mask = mask.detach()[0].cpu().numpy()
    mask = (mask * 255).round().astype("uint8")
    return Image.fromarray(mask)



'''
w direction settings
'''
pca = torch.load('./w_direction/gan_space/ffhq_pca.pt', map_location='cpu')
gan_space_configs = {}
with open('./w_direction/ganspace_configs.csv', "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        key = row.pop(0)
        gan_space_configs[key] = list(map(int, row))
gan_space_keys = list(gan_space_configs.keys())
w_direction_offline = torch.load('./w_direction/att_direction_scale.pth', map_location='cpu')


def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to(latent)#1*512
    lat_comp = pca['comp'].to(latent)#80*1*512
    lat_std = pca['std'].to(latent)#80
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta

def get_w_direction(latents_psp=None, max_strength=30, w_direction_offline=None, gan_space_configs=None, pca=None):
    edit_directions = []
    for latents_space in latents_psp: #bsz * 18 * 512
        direction_type = random.random()
        # strength = random.randint(-max_strength, max_strength)
        strength = random.uniform(-max_strength, max_strength)
        if direction_type > 0.4:
            direction = random.choice(w_direction_offline).squeeze(0) * strength  # 18*512
            direction = direction.to(latents_space)
        else:
            att_key = random.choice(list(gan_space_configs.keys()))
            select_config = gan_space_configs[att_key]
            pca_idx, start, end, _ = select_config 
            
            delta = get_delta(pca, latents_space, pca_idx, strength)
            delta_padded = torch.zeros(latents_space.shape).to(latents_space)
            delta_padded[start:end] += delta.repeat(end - start, 1)
            direction = delta_padded 
        edit_directions.append(direction) # 18*512
    edit_directions = torch.stack(edit_directions)
    return edit_directions

def print_networks(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params / 1e6




def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    
    #wplus-adapter
    w_proj_model = WProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        embeddings_dim=512,
        clip_extra_context_tokens=4,
    )

    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_wplus.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_wplus.weight": unet_sd[layer_name + ".to_v.weight"],
                "to_q_wplus.weight": unet_sd[layer_name + ".to_q.weight"],
            }
            attn_procs[name] = WPlusAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    wplus_adapter = WPlusAdapter(unet, w_proj_model, adapter_modules)


    
    '''
    ##load the w_proj parameters from pre-trained models
    '''
    if args.stage1_pretrain is not None:
        wplus_state = torch.load(args.stage1_pretrain, map_location="cpu")
        wplus_adapter.w_proj_model.load_state_dict(wplus_state["w_proj"])
        wplus_adapter.adapter_modules.load_state_dict(wplus_state["wplus_adapter"])
    '''
    Fixed during Stage 2
    '''
    wplus_adapter.w_proj_model.requires_grad_(False)


    if accelerator.is_main_process:
        accelerator.init_trackers("inv", config=vars(args))
        print(['WProjModel contains: {} M parameters'.format(print_networks(w_proj_model))])
        print(['Trainable w-attention contains: {} M parameters'.format(print_networks(adapter_modules))])

        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(wplus_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    wplus_adapter, optimizer, train_dataloader = accelerator.prepare(wplus_adapter, optimizer, train_dataloader)
    

    '''
    load all the parameters from pre-trained models
    '''
    # accelerator.load_state(path of pytorch_model.bin)
    # accelerator.print('Successfully load the pre-trained models')


    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(wplus_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["wild_images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor #4, 64, 64

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            

                image_embeds = batch['id_noises'][:,0,:,:] # for w+ B*18*512
   
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                image_embeds_negtive_ = []
                ind = list(range(image_embeds.size(0)))
                random.shuffle(ind)
                image_embeds_shuffle = image_embeds.detach().clone()
                image_embeds_shuffle = image_embeds_shuffle[ind,...]
                image_embeds_ori = image_embeds.detach()

                disturbance_type = random.uniform(0.0, 1.0)
                if disturbance_type > 0.55:
                    image_embeds_shuffle = image_embeds_shuffle + (image_embeds_ori - image_embeds_shuffle) * random.uniform(-40, 40)
                elif disturbance_type > 0.48:
                    image_embeds_shuffle = image_embeds_shuffle + image_embeds_ori * random.randint(-60, 60)/10.0
                elif disturbance_type > 0.35:
                    image_embeds_shuffle = image_embeds_shuffle + torch.normal(image_embeds.mean(), image_embeds.std(), image_embeds.size()).to(image_embeds) * random.randint(-80, 80)/10.0
                elif disturbance_type > 0.2: #using w direction augmentation
                    image_embeds_shuffle = image_embeds_ori + get_w_direction(image_embeds_ori, 60, w_direction_offline, gan_space_configs, pca)
                elif disturbance_type > 0.0:
                    direction1 = get_w_direction(image_embeds_ori, 60, w_direction_offline, gan_space_configs, pca)
                    direction2 = get_w_direction(image_embeds_ori, 60, w_direction_offline, gan_space_configs, pca)
                    alpha = random.uniform(0.0, 1.0) #random.random()
                    image_embeds_shuffle = image_embeds_ori + alpha * direction1 + (1 - alpha) * direction2

                for image_embed, drop_image_embed in zip(image_embeds_shuffle, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_negtive_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_negtive_.append(image_embed)
                image_embeds_negtive = torch.stack(image_embeds_negtive_)
            
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                
                noise_pred = wplus_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)
                
                with torch.no_grad():
                    noise_pred_augmentation = wplus_adapter(noisy_latents.detach(), timesteps, encoder_hidden_states.detach(), image_embeds_negtive.detach())
                    accelerator.unwrap_model(wplus_adapter).set_scale(0.0)
                    noise_pred_now = wplus_adapter(noisy_latents.detach(), timesteps, encoder_hidden_states.detach(), image_embeds_negtive.detach())
                    accelerator.unwrap_model(wplus_adapter).set_scale(1.0)

                mask_region = 1 - F.interpolate(batch['wild_masks'], (64, 64), mode='bilinear').repeat(1,4,1,1)

                loss_disen = F.mse_loss(noise_pred_augmentation.detach().float() * mask_region, noise_pred.float() * mask_region, reduction="mean")
                # loss_preserve = F.mse_loss(noise_pred_now.detach().float() * mask_region, noise_pred.float() * mask_region, reduction="mean")
                loss_preserve = F.mse_loss(noise_pred_now.detach().float(), noise_pred.float(), reduction="mean")
                loss_rec = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss_rec + loss_disen * args.lambda_disen + loss_preserve * args.lambda_preserve
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    # print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {:.5f}, rec_loss: {:.5f}, mask_loss: {:.5f}, preserve_loss: {:.5f}".format(
                    #     epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, loss_rec.detach().item(), loss_disen.detach().item(), loss_preserve.detach().item()))
                    print("Epoch {}, step {}, global step {}, step_loss: {:.5f}, rec_loss: {:.5f}, disen_loss: {:.5f}, preserve_loss: {:.5f}, lambda_disen {}, lambda_preserve {}, lr {}".format(
                        epoch, step, global_step, avg_loss, loss_rec.detach().item(), loss_disen.detach().item(), loss_preserve.detach().item(), args.lambda_disen, args.lambda_preserve, args.learning_rate))
            
            global_step += 1
            if accelerator.is_main_process:
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    accelerator.print('saving checkpoint {}'.format(global_step))
                    
            logs = {"loss": loss.detach().item(), "rec_loss": loss_rec.detach().item(), "disen_loss": loss_disen.detach().item(), "preserve_loss": loss_preserve.detach().item(), "lr": args.learning_rate, 'lr_proj':args.learning_rate_proj, 'lambda_disen': args.lambda_disen, 'lambda_preserve': args.lambda_preserve}
            accelerator.log(logs, step=global_step)

            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    

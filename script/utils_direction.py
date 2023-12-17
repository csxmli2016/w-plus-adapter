
import torch
import os
from PIL import Image
import numpy as np
import csv

'''
\Delta w direction
'''
def get_delta(pca, latent, idx, strength):
    # pca: ganspace checkpoint. latent: (16, 512) w+
    w_centered = latent - pca['mean'].to(latent)#1*512
    lat_comp = pca['comp'].to(latent)#80*1*512
    lat_std = pca['std'].to(latent)#80
    w_coord = torch.sum(w_centered[0].reshape(-1)*lat_comp[idx].reshape(-1)) / lat_std[idx]
    delta = (strength - w_coord)*lat_comp[idx]*lat_std[idx]
    return delta



def get_direction_from_ganspace(att_key='displeased', latents_psp=None, strength=10):
    pca = torch.load('./w_direction/gan_space/ffhq_pca.pt')
    gan_space_configs = {}
    with open('./w_direction/gan_space/ganspace_configs.csv', "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            key = row.pop(0)
            gan_space_configs[key] = list(map(int, row))

    latents_space = latents_psp[0]
    select_config = gan_space_configs[att_key]
    pca_idx, start, end, _ = select_config 
    
    delta = get_delta(pca, latents_space, pca_idx, strength)
    delta_padded = torch.zeros(latents_space.shape).to(latents_space)
    delta_padded[start:end] += delta.repeat(end - start, 1)
    direction = delta_padded.unsqueeze(0)

    return direction
    
    

def get_direction_from_interfacegan(att_key='age'):
    direction = torch.load('./w_direction/interfacegan_directions/{}.pt'.format(att_key), map_location='cpu')
    if direction.dim() == 2:
        direction = direction.unsqueeze(0).repeat(1,18,1)
    return direction


def get_direction_from_latentdirection(att_key='age'):
    direction = np.load('./w_direction/latent_directions/{}.npy'.format(att_key))
    direction = torch.from_numpy(direction)
    direction = direction.unsqueeze(0)
    return direction



'''
Other tools
'''
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.
    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
from utils_direction import load_file_from_url
import os.path as osp

# pretrained w-plus adapter
if not osp.exists('./pretrain_models/wplus_adapter.bin'):
    download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/wplus_adapter.bin'
    load_file_from_url(url=download_url, model_dir='./pretrain_models/', progress=True, file_name=None)


# pretrained e4e and stylegan
if not osp.exists('./script/weights/e4e_ffhq_encode.pt'):
    download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/e4e_ffhq_encode.pt'
    load_file_from_url(url=download_url, model_dir='./script/weights/', progress=True, file_name=None)


if not osp.exists('./script/weights/mmod_human_face_detector.dat'):
    download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/mmod_human_face_detector.dat'
    load_file_from_url(url=download_url, model_dir='./script/weights/', progress=True, file_name=None)

if not osp.exists('./script/weights/psfrgan_epoch15_net_G.pth'):
    download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/psfrgan_epoch15_net_G.pth'
    load_file_from_url(url=download_url, model_dir='./script/weights/', progress=True, file_name=None)


if not osp.exists('./script/weights/shape_predictor_68_face_landmarks-fbdc2cb8.dat'):
    download_url = 'https://github.com/csxmli2016/w-plus-adapter/releases/download/v1/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
    load_file_from_url(url=download_url, model_dir='./script/weights/', progress=True, file_name=None)
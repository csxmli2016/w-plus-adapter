
import os
import os.path as osp
import numpy as np

from PIL import Image
import scipy
import scipy.ndimage
import argparse
from utils import load_file_from_url, align_face, color_parse_map, pil2tensor, tensor2pil

import torchvision.transforms as transforms

from models.parsenet import ParseNet
import torch
import cv2


try:
    import dlib
except ImportError:
    print('Please install dlib by running:' 'conda install -c conda-forge dlib')



def get_landmark(filepath, detector, predictor):
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    if len(dets) == 0:
        return None

    all_lm = []

    for det in dets:
        if isinstance(detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det

        shape = predictor(img, rec) 

        single_points = []
        for s in list(shape.parts()):
            single_points.append([s.x, s.y])
        all_lm.append(np.array(single_points))

    if len(all_lm) <= 0:
        return None
    else:
        return all_lm
    


'''
For in the wild face image: CUDA_VISIBLE_DEVICES=0 python ./script/ProcessWildImage.py -i ./test_data/in_the_wild -o ./test_data/in_the_wild_Result -n -s
For aligned face image: CUDA_VISIBLE_DEVICES=0 python ./script/ProcessWildImage.py -i ./test_data/aligned_face -o ./test_data/aligned_face_Result -s
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='./inputs/whole_imgs')
    parser.add_argument('-o', '--out_dir', type=str, default='./inputs/cropped_faces')
    parser.add_argument('-m', '--min_size', type=int, default=160)
    parser.add_argument('-n', '--need_alignment', action='store_true', help='input face image needs alignment like FFHQ')
    parser.add_argument('-c', '--cnn_detector', action='store_true', help='do not use cnn face detector in dlib.')
    parser.add_argument('-s', '--sr', action='store_true', help='using blind face restoration method')
    args = parser.parse_args()
    
    '''
    Step 1: Face Crop and Alignment
    '''
    img_list = os.listdir(args.in_dir)
    img_list.sort()
    test_img_num = len(img_list)

    if args.out_dir.endswith('/'):  # solve when path ends with /
        args.out_dir = args.out_dir[:-1]
    
    save_path_step1 = args.out_dir + '-Step1-AlignedFace'
    os.makedirs(save_path_step1, exist_ok=True)

    if args.need_alignment:
        ##using dlib for landmark detection
        os.makedirs('./script/weights', exist_ok=True)
        if not osp.exists('./script/weights/shape_predictor_68_face_landmarks-fbdc2cb8.dat'):
            shape_predictor_url = 'https://github.com/csxmli2016/csxmli2016.github.io/releases/download/1.0/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
            ckpt_path = load_file_from_url(url=shape_predictor_url, model_dir='./script/weights', progress=True, file_name=None)
        predictor = dlib.shape_predictor('./script/weights/shape_predictor_68_face_landmarks-fbdc2cb8.dat')

        if args.cnn_detector:
            if not osp.exists('./script/weights/mmod_human_face_detector.dat'):
                cnn_predictor_url = 'https://github.com/chaofengc/PSFRGAN/releases/download/v0.1.0/mmod_human_face_detector.dat'
                ckpt_path = load_file_from_url(url=cnn_predictor_url, model_dir='./script/weights', progress=True, file_name=None)
            detector = dlib.cnn_face_detection_model_v1('./script/weights/mmod_human_face_detector.dat')
        else:
            detector = dlib.get_frontal_face_detector()

        for i, in_name in enumerate(img_list):
            in_path = osp.join(args.in_dir, in_name)
            img_name = os.path.basename(in_path)

            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')

            try:
                all_lm = get_landmark(in_path, detector, predictor)
                if all_lm is None:
                    print('No landmark for image: {}. Continue... '.format(in_path))
                    continue
            except:
                print('Detected Error for image: {}. Continue... '.format(in_path))
                continue
            for idx, lm in enumerate(all_lm):
                out_name = '{}_{}.png'.format(osp.splitext(img_name)[0], idx)
                out_path = os.path.join(save_path_step1, out_name)  

                img, length = align_face(in_path, lm)
                if length > args.min_size:
                    img.save(out_path)
                else:
                    print('{} contains face image with lower size (<={}), we do not save it...'.format(img_name, args.min_size))
                # exit('ss')
        
        print('#'*10 + 'Step 1: Align and Crop Face Done!' + '#'*10)
    else:
        for i, in_name in enumerate(img_list):
            in_path = osp.join(args.in_dir, in_name)
            img_name = os.path.basename(in_path)
            img = Image.open(in_path)
            img = img.resize((512,512), Image.BILINEAR)
            out_name = '{}.png'.format(osp.splitext(img_name)[0])
            out_path = os.path.join(save_path_step1, out_name)  
            img.save(out_path)
        print('#'*10 + 'Input face is aligned. Skip Step1...' + '#'*10)



    '''
    Step 2: Remove Background
    '''
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    parse_net = ParseNet(512, 512, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
    parse_net.eval()
    if not osp.exists('./script/weights/parse_multi_iter_90000.pth'):
        url = 'https://github.com/chaofengc/PSFRGAN/releases/download/v0.1.0/parse_multi_iter_90000.pth'
        ckpt_path = load_file_from_url(url=url, model_dir='./script/weights', progress=True, file_name=None)

    parse_net.load_state_dict(torch.load('./script/weights/parse_multi_iter_90000.pth'))


    if args.sr:
        from models.bfrnet import PSFRGenerator
        bfr_net = PSFRGenerator(3, 3, in_size=512, out_size=512, relu_type='LeakyReLU', parse_ch=19, norm_type='spade')
        for m in bfr_net.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
        bfr_net.eval()
        if not osp.exists('./script/weights/psfrgan_epoch15_net_G.pth'):
            url = 'https://github.com/chaofengc/PSFRGAN/releases/download/v0.1.0/psfrgan_epoch15_net_G.pth'
            ckpt_path = load_file_from_url(url=url, model_dir='./script/weights', progress=True, file_name=None)

        bfr_net.load_state_dict(torch.load('./script/weights/psfrgan_epoch15_net_G.pth'))

    
    img_step2_list = os.listdir(save_path_step1)
    img_step2_list.sort()


    save_path_step2 = args.out_dir + '-Step2-FaceMask'
    os.makedirs(save_path_step2, exist_ok=True)
    if args.sr:
        save_path_step2_sr = args.out_dir + '-Step2-BFR'
        os.makedirs(save_path_step2_sr, exist_ok=True)

    for i, img_name in enumerate(img_step2_list):
        print(f'[{i+1}/{len(img_step2_list)}] Processing: {img_name}')
        img = Image.open(osp.join(save_path_step1, img_name)).convert('RGB')
        img = img.resize((512, 512), Image.BILINEAR)
        
        img_tensor = trans(img).unsqueeze(0)
        with torch.no_grad():
            parse_map, _ = parse_net(img_tensor)
            if args.sr:
                parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
                output_SR = bfr_net(img_tensor, parse_map_sm)
                save_img_sr = tensor2pil(output_SR)
                save_img_sr.save(osp.join(save_path_step2_sr, img_name))

        parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
        parse_img = color_parse_map(parse_map_sm)

        img_np = parse_img[0]
        img_np = np.mean(img_np, axis=2)
        img_np[img_np>0] = 255
        img_np = 255 - img_np
        save_img = Image.fromarray(img_np.astype(np.uint8))
        save_img.save(osp.join(save_path_step2, img_name))

    print('#'*10 + 'Step 2: Face Segmentation Done!' + '#'*10)


    '''
    Step 3: Get w from Cropped Face
    '''
    from models.psp import pSp

    device = 'cuda'
    e4e_path = './script/weights/e4e_ffhq_encode.pt'
    if not osp.exists(e4e_path):
        url = 'https://github.com/csxmli2016/csxmli2016.github.io/releases/download/1.0/e4e_ffhq_encode.pt'
        ckpt_path = load_file_from_url(url=url, model_dir='./script/weights', progress=True, file_name=None)

    e4e_ckpt = torch.load(e4e_path, map_location='cpu')
    latent_avg = e4e_ckpt['latent_avg'].to(device)
    e4e_opts = e4e_ckpt['opts']
    e4e_opts['checkpoint_path'] = e4e_path
    e4e_opts['device'] = device
    opts = argparse.Namespace(**e4e_opts)
    e4e = pSp(opts).to(device)
    e4e.eval()

    if not args.sr:
        save_path_step3 = args.out_dir + '-Step3-e4e'
        save_path_step3_rec = args.out_dir + '-Step3-e4e-rec'
    else:
        save_path_step3 = args.out_dir + '-Step3-BFR-e4e'
        save_path_step3_rec = args.out_dir + '-Step3-BFR-e4e-rec'

    os.makedirs(save_path_step3, exist_ok=True)
    os.makedirs(save_path_step3_rec, exist_ok=True)


    for i, img_name in enumerate(img_step2_list):
        print(f'[{i+1}/{len(img_step2_list)}] Processing: {img_name}')

        if not args.sr:
            image = Image.open(os.path.join(save_path_step1, img_name)).convert('RGB')
        else:
            image = Image.open(os.path.join(save_path_step2_sr, img_name)).convert('RGB')
        image = pil2tensor(image)
        image = (image - 127.5) / 127.5     # Normalize
        
        mask = Image.open(os.path.join(save_path_step2, img_name))

        kernel_size = 5
        mask_image = cv2.GaussianBlur(np.array(mask), (kernel_size, kernel_size), 0)
        mask_image = Image.fromarray(mask_image.astype(np.uint8)) 

        mask_image = mask_image.resize((256, 256))
        mask_image = np.asarray(mask_image).astype(np.float32) # C,H,W -> H,W,C
        mask_image = torch.FloatTensor(mask_image.copy())
        mask = mask_image / 255.0
        image = image * (1 - mask) + mask

        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            latents_psp = e4e.encoder(image)
        if latents_psp.ndim == 2:
            latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)[:, 0, :]
        else:
            latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)

        torch.save(latents_psp, osp.join(save_path_step3, img_name[:-4]+'.pth'))
        if i < 100:
            with torch.no_grad():
                imgs, _ = e4e.decoder([latents_psp[0].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
            imgs_of = tensor2pil(imgs)
            imgs_of = imgs_of.resize((256,256))
            imgs_of.save(osp.join(save_path_step3_rec, img_name[:-4]+'.jpg'))

    print('#'*10 + 'Step 3: Get w Done! You can check the e4e reconstruction from {}'.format(save_path_step3_rec) + '#'*10)

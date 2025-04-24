#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from torchvision.transforms import Compose, Lambda
from utils.dtypes import LabelEnum
import nibabel as nib
import torchio as tio
import numpy as np
import argparse
import torch
import os
import glob


exportfolder ='../../../result/generator/Normal_DWI'
inputfolder = '../../../result/generator/Normal_mask'
input_size = 128
depth_size = 64
batchsize = 10
weightfile = '../../../model/med_ddpm/dwi_250.pt'
num_channels = 64
num_res_blocks = 1
num_samples = 1
in_channels =3
out_channels = 1
device = "cuda:5" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
i=2
mask_list = sorted(glob.glob(f"{inputfolder}/*.nii.gz"))
print(len(mask_list))


def resize_img_4d(input_img):
    d, h, w, c = input_img.shape  # Axial 기준: (D, H, W, C)
    result_img = np.zeros((depth_size, input_size, input_size, in_channels - 1))  # (D, H, W, C)

    if d != depth_size or h != input_size or w != input_size:
        for ch in range(c):
            buff = input_img[..., ch]  # (D, H, W)
            img = tio.ScalarImage(tensor=buff[np.newaxis, ...])  # (1, D, H, W)
            cop = tio.Resize((depth_size, input_size, input_size))
            img = np.asarray(cop(img))[0]  # (D, H, W)
            result_img[..., ch] = img
        return result_img
    else:
        return input_img


def label2masks(masked_img):
    result_img = np.zeros(masked_img.shape + (in_channels-1,))
    result_img[masked_img==LabelEnum.BRAINAREA.value, 0] = 1
    result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
    return result_img


input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.unsqueeze(0))
])

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).to(device)


diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = 250,   # number of steps
    loss_type = 'L1', 
    with_condition=True,
).to(device)
diffusion.load_state_dict(torch.load(weightfile,map_location=device)['ema'])
print("Model Loaded!")

# +
img_dir = exportfolder


max_file_size_kb = 3400
max_retry = 10
img_dir = exportfolder
for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k + 1)
    print("LEFT: ", left)

    ref = nib.load(inputfile)
    msk_name = inputfile.split('/')[-1]
    refImg = ref.get_fdata()
    img = label2masks(refImg)
    img = resize_img_4d(img)
    input_tensor = input_transform(img)
    condition_tensor = input_tensor.to(device)

    for sample_idx in range(num_samples):
        saved_count = 0
        retry = 0
        file_saved = False
        generated = diffusion.sample(batch_size=batchsize, condition_tensors=condition_tensor.repeat(batchsize, 1, 1, 1, 1))
        generated = generated.unsqueeze(1).cpu().numpy()  # (B, 1, D, H, W)

        for b in range(batchsize):
            sampleImage = generated[b][0]  # shape: (D, H, W)
            sampleImage = sampleImage.reshape(refImg.shape)

            # 저장 경로 생성
            out_name = f"{msk_name}"
            nifti_path = os.path.join(img_dir, out_name)
            nifti_img = nib.Nifti1Image(sampleImage, affine=ref.affine)
            nib.save(nifti_img, nifti_path)
            file_size_kb = os.path.getsize(nifti_path) / 1024
            if file_size_kb <= max_file_size_kb:
                file_saved = True
                saved_count += 1
                break
            else:
                os.remove(nifti_path)  # 너무 크면 삭제하고 재시도
                continue
    
    torch.cuda.empty_cache()

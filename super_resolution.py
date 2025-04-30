import torch
from RealESRGAN import RealESRGAN
import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision.utils import save_image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import random
from PIL import Image
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from matplotlib import pyplot as plt
import nibabel as nib
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda",1)
print(f"Device:\t\t{device}")
import pytorch_model_summary as tms
import random


params={'image_size':256,
        'lr':1e-4,
        'beta1':0.5,
        'beta2':0.999,
        'batch_size':32,
        'epochs':1000,
        'n_classes':None,
        'data_path':'../../data/SR_dataset/SWI/',
        'image_count':10000,
        'inch':3,
        'modch':64,
        'outch':3,
        'chmul':[1,2,4,8],
        'numres':2,
        'dtype':torch.float32,
        'cdim':10,
        'useconv':False,
        'droprate':0.1,
        'T':1000,
        'w':1.8,
        'v':0.3,
        'multiplier':2.5,
        'threshold':0.1,
        'ddim':True,
        }

topilimage = torchvision.transforms.ToPILImage()
tf=transforms.ToTensor()
def transback(data:Tensor) -> Tensor:
        data=data-data.min()
        data=data/data.max()

        return data
    
    

# 예시 데이터셋 클래스 (HR, LR 이미지 쌍)
class SRDataset(Dataset):
    def __init__(self, hr_imgs, scale=2):
        self.scale = scale
        self.hr_images = hr_imgs
    def __len__(self):
        return len(self.hr_images)
    
    def trans(self,image):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)
            
        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)
            
        return image
    def random_resample(self,image, scale_factor=0.5):
        # 사용할 보간 방법 리스트
        image=topilimage(image).convert("RGB")
        resampling_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS
        ]

        # 랜덤하게 보간 방법 선택
        chosen_method = random.choice(resampling_methods)

        # 새로운 크기 계산
        new_size = (
            int(image.width * scale_factor),
            int(image.height * scale_factor)
        )

        # 이미지를 새로운 크기로 변경
        image1 = image.resize(new_size, resample=chosen_method)
        image1=tf(image1)
        image=tf(image)
        return image,image1
    
    def __getitem__(self, index):
        hr_image,lr_image = self.random_resample(self.hr_images[index], scale_factor=0.5)
        return lr_image, hr_image
    
nii_list=glob(params['data_path']+'*.nii.gz')
train_img=torch.zeros((len(nii_list)*20,1,params['image_size'],params['image_size']),dtype=torch.float32)
for i in tqdm(range(len(nii_list))):
    nii_img=nib.load(nii_list[i])
    nii_img=nii_img.get_fdata()
    nii_img=(nii_img.astype(np.float32)+1.)/2.
    random_integers =random.sample(range(0, nii_img.shape[0]), 20)
    for j, index in enumerate(random_integers):
        train_img[i*20+j]=torch.from_numpy(nii_img[index]).unsqueeze(0)

split=int(len(train_img)*0.95)
train_img1=train_img[:split]
test_img=train_img[split:]
train_dataset=SRDataset(train_img)
test_dataset=SRDataset(test_img)
dataloader=DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True,drop_last=True)
test_dataloader=DataLoader(test_dataset,batch_size=params['batch_size'],shuffle=True)


class CombinedLoss(nn.Module):
    def __init__(self, device, vgg_weight=0.006, l1_weight=1.0, adv_weight=0.001):
        super(CombinedLoss, self).__init__()
        
        # L1 Loss
        self.l1_loss = nn.L1Loss()

        # VGG19 for Perceptual Loss
        vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        # Weights for each loss component
        self.vgg_weight = vgg_weight
        self.l1_weight = l1_weight
        self.adv_weight = adv_weight

    def perceptual_loss(self, fake_img, real_img):
        fake_features = self.vgg(fake_img)
        real_features = self.vgg(real_img)
        return F.l1_loss(fake_features, real_features)

    def forward(self, fake_img, real_img, disc_fake_pred=None):
        # L1 Loss
        l1_loss = self.l1_loss(fake_img, real_img)
        
        # Perceptual Loss
        perceptual_loss = self.perceptual_loss(fake_img, real_img)
        
        # Adversarial Loss (if provided)
        if disc_fake_pred is not None:
            adversarial_loss = F.softplus(-disc_fake_pred).mean()
        else:
            adversarial_loss = 0.0

        # Combined loss with weights
        total_loss = (
            self.l1_weight * l1_loss +
            self.vgg_weight * perceptual_loss +
            self.adv_weight * adversarial_loss
        )
        
        return total_loss
    
model = RealESRGAN(device, scale=2).model.to(device)
criterion =CombinedLoss(device, vgg_weight=0.2, l1_weight=1.0, adv_weight=0.1)
optimizer = optim.Adam(model.parameters(), lr=params['lr'], betas=(params['beta1'], params['beta2']))
model.load_state_dict(torch.load('../../model/ESRGAN/SWI/ckpt_762_checkpoint.pt',map_location=device))
topilimage = torchvision.transforms.ToPILImage()
scaler = torch.cuda.amp.GradScaler()
for epc in range(params['epochs']):
    model.train()
    total_loss=0
    steps=0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for lr_image, hr_image in tqdmDataLoader:
            ir= lr_image.to(device)
            hr= hr_image.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(ir), hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps+=1
            total_loss+=loss.item()
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss: ": total_loss/steps,
                    "batch per device: ":ir.shape[0],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
            )

    model.eval()
    with torch.no_grad():
        # 첫 배치 기준, 또는 무작위 샘플 가져오기
        lr_sample, hr_sample = next(iter(test_dataloader))
        lr_sample = lr_sample.to(device)
        hr_sample = hr_sample.to(device)

        # 모델 추론
        sr_sample = model(lr_sample)

        # 첫 번째 샘플만 PIL로 변환해서 시각화
        def to_numpy_img(tensor):
            tensor = tensor.detach().cpu().clamp(0, 1)  # ensure valid range
            return tensor.permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(to_numpy_img(lr_sample[0]))
        axs[0].set_title("Low Resolution")
        axs[0].axis('off')

        axs[1].imshow(to_numpy_img(sr_sample[0]))
        axs[1].set_title("Super Resolved")
        axs[1].axis('off')

        axs[2].imshow(to_numpy_img(hr_sample[0]))
        axs[2].set_title("High Resolution (GT)")
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'../../result/ESRGAN/SWI/{epc+1}_SR.png', dpi=600)
        plt.close()

        torch.save(model.state_dict(), f'../../model/ESRGAN/SWI/ckpt_{epc+1}_checkpoint.pt')
    torch.cuda.empty_cache()
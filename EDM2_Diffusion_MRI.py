import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import torch.optim as optim
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch import Tensor
from PIL import Image
from torchinfo import summary
from EDM2.model import UNet, EDM2Wrapper, edm2_loss, edm2_sample


class_list = ['DWI', 'ADC']
params = {
    'image_size': 256,
    'lr': 1e-4,
    'batch_size': 64,
    'epochs': 10000,
    'data_path':'../../data/2D_MRI/',
    'image_count': 5000,
    'inch': 1,
    'outch': 1,
    'cdim': 10,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'threshold': 0.02,
    'save_every': 5,
    'save_path': '/edm2/MRI',
    'rho': 7,
    'S_churn': 0,
    'S_noise': 1.0
}

print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda", 1)
print(f"Device:\t\t{device}")

# 2. Transform 및 역정규화 함수
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ⚙️ 전처리 함수
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
])

def transback(data: torch.Tensor) -> torch.Tensor:
    return data * 0.5 + 0.5  # → [0, 1]

def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# 📂 이미지 경로 및 라벨 수집
image_paths, image_labels = [], []
for i, cname in enumerate(class_list):
    paths = sorted(glob(os.path.join(params['data_path'], cname, '*.png')))[:params['image_count']]
    image_paths.extend(paths)
    image_labels.extend([i] * len(paths))

# 🖼️ 이미지를 메모리에 Tensor로 로딩 (흑백 1채널 기준)
N = len(image_paths)
C, H, W = params['inch'], params['image_size'], params['image_size']
train_images = torch.zeros((N, C, H, W))

print("Loading images into tensor...")
for i, path in enumerate(tqdm(image_paths)):
    img = Image.open(path).convert('L').resize((W, H))
    train_images[i] = trans(img)

train_labels = torch.tensor(image_labels, dtype=torch.long)

# 🧱 Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        lab = self.labels[index]

        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
        if random.random() > 0.5:
            img = transforms.functional.vflip(img)

        return img, lab

    def __len__(self):
        return len(self.images)

# 📦 최종 Dataset & DataLoader
train_dataset = CustomDataset(train_images, train_labels)
dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)


unet = UNet(params['inch'], params['outch'], cond_dim=params['cdim']).to(device)
edm2_model = EDM2Wrapper(unet, sigma_min=params['sigma_min'], sigma_max=params['sigma_max']).to(device)
optimizer = optim.AdamW(edm2_model.parameters(), lr=params['lr'], weight_decay=1e-4)

# 6. 모델 구조 요약 출력
test_batch = 1
image_input = torch.randn(test_batch, 1, params['image_size'], params['image_size']).to(device)
sigma_input = torch.ones(test_batch, 1).to(device) * 10.0
class_input = torch.randint(0, len(class_list), (test_batch,)).to(device)

summary(edm2_model.model, 
        input_data=(image_input, sigma_input, unet.get_condition_embedding(class_input)),
        col_names=["input_size", "output_size", "num_params"])

for epc in range(params['epochs']):
    edm2_model.train()
    total_loss = 0
    steps = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()

            if random.random() < params['threshold']:
                mask = torch.rand(lab.shape[0]) < 0.5  # 절반만 mask
                lab[mask] = -1  # -1 or special token

            loss = edm2_loss(edm2_model, img, lab)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            tqdmDataLoader.set_postfix({
                'epoch': epc + 1,
                'loss': total_loss / steps,
                'lr': optimizer.param_groups[0]['lr']
            })

    # 8. 샘플 생성 및 모델 저장
    if epc % params['save_every'] == 0:
        edm2_model.eval()
        with torch.no_grad():
            each_device_batch = params['batch_size'] // len(class_list)
            lab = torch.arange(len(class_list)).repeat(each_device_batch).to(device)
            genshape = (len(lab), params['outch'], params['image_size'], params['image_size'])
            samples = edm2_sample(
                edm2_model, genshape, 
                num_steps=18,
                sigma_min=params['sigma_min'],
                sigma_max=params['sigma_max'],
                rho=params['rho'],
                S_churn=params['S_churn'],
                S_noise=params['S_noise'],
                class_labels=lab
            )
            samples = transback(samples)

        result_path = '../../result' + params['save_path']
        model_path = '../../model' + params['save_path']
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        save_image(samples, f'{result_path}/generated_{epc+1}_pict.png', nrow=each_device_batch)
        torch.save({
            'model': edm2_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'{model_path}/ckpt_{epc+1}.pt')

        torch.cuda.empty_cache()
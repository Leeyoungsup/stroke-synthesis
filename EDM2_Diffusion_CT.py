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
from torchinfo import summary
from PIL import Image
from EDM2.edm2Diffutsion import UNet, EDM2Wrapper, edm2_loss

# 클래스 정의
class_list = ['Normal', 'Hemorrhagic']

# 하이퍼파라미터 설정
params = {
    'image_size': 64,
    'lr': 2e-5,
    'batch_size': 64,
    'epochs': 20000,
    'data_path': '../../data/2D_CT/',
    'image_count': 5000,
    'inch': 1,
    'outch': 1,
    'cdim': len(class_list),

    'sigma_min': 0.01,
    'sigma_max': 80.0,
    'rho': 3,
    'S_churn': 40,
    'S_noise': 1.0,

    'threshold': 0.0,
    'save_every': 5,
    'save_path': '/edm2/CT'
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 변환 정의
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def transback(data: torch.Tensor) -> torch.Tensor:
    return data * 0.5 + 0.5

# 이미지 로드
image_paths, image_labels = [], []
for i, cname in enumerate(class_list):
    paths = sorted(glob(os.path.join(params['data_path'], cname, '*.png')))[:params['image_count']]
    image_paths.extend(paths)
    image_labels.extend([i] * len(paths))

N = len(image_paths)
C, H, W = params['inch'], params['image_size'], params['image_size']
train_images = torch.zeros((N, C, H, W))

print("Loading images into tensor...")
for i, path in enumerate(tqdm(image_paths)):
    img = Image.open(path).convert('L').resize((W, H))
    train_images[i] = trans(img)

train_labels = torch.tensor(image_labels, dtype=torch.long)

# 커스텀 Dataset
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

# DataLoader
train_dataset = CustomDataset(train_images, train_labels)
dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

# 모델 초기화
unet = UNet(params['inch'], base_ch=64, ch_mults=(1, 2, 4), emb_dim=256, cond_dim=params['cdim']).to(device)
edm2_model = EDM2Wrapper(unet, len(class_list), sigma_min=params['sigma_min'], sigma_max=params['sigma_max'], rho=params['rho']).to(device)
optimizer = optim.AdamW(edm2_model.parameters(), lr=params['lr'], weight_decay=1e-4)

# 모델 요약 출력
image_input = torch.randn(4, 1, params['image_size'], params['image_size']).to(device)
sigma_input = torch.ones(4, 1).to(device) * 10.0
class_input = torch.randint(0, len(class_list), (4,)).to(device)
summary(edm2_model.model, input_data=(image_input, sigma_input, unet.get_condition_embedding(class_input, num_classes=len(class_list))), col_names=["input_size", "output_size", "num_params"])

for epc in range(params['epochs']):
    edm2_model.train()
    total_loss = 0
    steps = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()

            if random.random() < params['threshold']:
                mask = torch.rand(lab.shape[0]) < 0.5
                lab[mask] = -1

            loss = edm2_loss(edm2_model, img, lab, sigma_data=0.5,num_class=len(class_list))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            tqdmDataLoader.set_postfix({
                'epoch': epc + 1,
                'loss': total_loss / steps,
                'lr': optimizer.param_groups[0]['lr']
            })

    # ------------------------
    # Sample & Save
    # ------------------------
    if epc % params['save_every'] == 0:
        edm2_model.eval()
        with torch.no_grad():
            each_device_batch = params['batch_size'] // len(class_list)
            lab = torch.arange(len(class_list)).repeat(each_device_batch).to(device)
            genshape = (len(lab), params['outch'], params['image_size'], params['image_size'])
            samples = edm2_model.sample(
                shape=genshape,
                num_steps=50,
                S_churn=params['S_churn'],
                S_min=params['sigma_min'],
                S_max=params['sigma_max'],
                S_noise=params['S_noise'],
                guidance_weight=2.0,
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
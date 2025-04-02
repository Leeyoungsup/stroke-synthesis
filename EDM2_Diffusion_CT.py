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
from edm2_pytorch.model import SongUNet, DhariwalUNet, VPPrecond, VEPrecond, iDDPMPrecond, EDMPrecond


# 클래스 정의
class_list = ['Normal','Ischemic','Hemorrhagic']

params = {
    # 데이터 설정
    'data_path': '../../data/2D_CT/',
    'image_count': 10000,
    'image_size': 256,
    'inch': 1,
    'outch': 1,

    # 학습 설정
    'lr': 2e-4,
    'batch_size': 8,
    'epochs': 10000,
    'save_every': 10,
    'save_path': '../../result/edm2/CT',

    # EDM 샘플링 관련
    'P_mean': -1.2,
    'P_std': 1.2,
    'rho': 7.0,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'sigma_data': 0.5,
    'threshold': 0.0,

    # 모델 구조
    'cdim': 64,                        # base channels
    'channel_mult': [1, 2, 4, 8],      # 채널 증가 비율
    'attn_resolutions': [32,16,8],           # self-attention이 들어갈 해상도 (예: [16])
    'layers_per_block': 2           # 각 레벨마다 residual block 수
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 변환 정의
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
# ])
trans = transforms.Compose([
    transforms.ToTensor()
])
def transback(x):
    return (x.clamp(-1, 1) + 1) * 0.5

# 이미지 로드
image_paths, image_labels = [], []
for i, cname in enumerate(class_list):
    paths = sorted(glob(os.path.join(params['data_path'], cname, '*.png')))[:params['image_count']]
    image_paths.extend(paths)
    image_labels.extend([i] * len(paths))

N = len(image_paths)
C, H, W = params['inch'], params['image_size'], params['image_size']
train_images = torch.zeros((N, C, H, W), dtype=torch.float32)

print("Loading images into tensor...")
for i, path in enumerate(tqdm(image_paths)):
    img = Image.open(path).convert('L').resize((W, H))
    
    train_images[i] = trans(img)
train_images=train_images*2-1.
train_labels = torch.tensor(image_labels, dtype=torch.long)

# 커스텀 Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        lab = self.labels[index]
        # if random.random() > 0.5:
        #     img = transforms.functional.hflip(img)
        # if random.random() > 0.5:
        #     img = transforms.functional.vflip(img)
        return img, lab

    def __len__(self):
        return len(self.images)

# DataLoader
train_dataset = CustomDataset(train_images, train_labels)
dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,drop_last=True)
# 모델 초기화
model = EDMPrecond(
    img_resolution=params['image_size'],
    img_channels=params['inch'],
    label_dim=len(class_list),
    use_fp16=False,
    sigma_min=params['sigma_min'],
    sigma_max=params['sigma_max'],
    sigma_data=params['sigma_data'],
    model_type='DhariwalUNet',  # 또는 'SongUNet'
    model_channels=params['cdim'],
    channel_mult=params['channel_mult'],
    channel_mult_emb=4,
    num_blocks=params['layers_per_block'],
    attn_resolutions=params['attn_resolutions'],
    dropout=0.1,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=params['lr'])
#모델 불러오기
# model.load_state_dict(torch.load('../../model/edm2/CT/model_epoch_41.pt'))
# 모델 요약
summary(
    model,
    input_data=(
        torch.randn(1, params['inch'], params['image_size'], params['image_size']).to(device),  # noised input
        torch.tensor([params['sigma_data']], device=device),  # sigma
        torch.nn.functional.one_hot(torch.tensor([0]), num_classes=len(class_list)).float().to(device)  # dummy class label
    ),
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    depth=4,
    verbose=1
)
scaler = torch.cuda.amp.GradScaler()
for epoch in range(1, params['epochs'] + 1):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{params['epochs']}")
    for step, (imgs, labels) in enumerate(pbar, start=1):
        imgs, labels = imgs.to(device), labels.to(device)

        # EDM 논문 공식 σ 샘플링 방식 (log-normal)
        rnd_normal = torch.randn([imgs.shape[0]], device=imgs.device)
        sigmas = (params['sigma_data'] ** 2 + (rnd_normal * params['P_std'] + params['P_mean']).exp() ** 2).sqrt()

        # 노이즈 추가
        noise = torch.randn_like(imgs)
        noised = imgs + sigmas.view(-1, 1, 1, 1) * noise

        # 클래스 one-hot encoding
        class_onehot = torch.nn.functional.one_hot(labels, num_classes=len(class_list)).float()

        # 모델 forward 및 손실 계산
        
        denoised = model(noised, sigmas, class_labels=class_onehot)
        target = imgs
        l1 = (denoised - target).abs().mean()
        l2 = torch.nn.functional.mse_loss(denoised, target)
        loss = 0.8 * l1 + 0.2 * l2

        # 역전파 및 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / step
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    
        # 주기적으로 샘플 저장
    if (epoch - 1) % params['save_every'] == 0:
        model.eval()
        with torch.no_grad():
            # 각 클래스별 동일 개수로 label 생성
            num_per_class = params['batch_size'] // len(class_list)
            label_list = []
            for i in range(len(class_list)):
                label_list.extend([i] * num_per_class)
            label_tensor = torch.tensor(label_list, device=device)
            class_onehot = torch.nn.functional.one_hot(label_tensor, num_classes=len(class_list)).float()

            # 입력 noise 생성
            z = torch.randn(len(label_tensor), params['inch'], params['image_size'], params['image_size']).to(device)

            # 샘플링 루프: Euler-style
            sigma = torch.full((z.shape[0], 1, 1, 1), params['sigma_max'], device=device)
            for _ in tqdm(range(50), desc="Sampling"):
                denoised = model(z, sigma, class_labels=class_onehot)
                d = (z - denoised) / sigma
                dt = -0.9 * sigma
                z = z + d * dt
                sigma = sigma + dt
                sigma = sigma.clamp(min=params['sigma_min'])

            samples = transback(z)
            save_image(samples, os.path.join(params['save_path'], f'sample_epoch_{epoch}.png'), nrow=num_per_class)
            torch.save(model.state_dict(), os.path.join(params['save_path'].replace('result','model'), f'model_epoch_{epoch}.pt'))

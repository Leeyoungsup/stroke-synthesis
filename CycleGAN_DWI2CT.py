# 기본 PyTorch 관련 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

# torchvision 및 기타 이미지 처리 관련
from torchvision import transforms, datasets
from torchvision.utils import save_image
from PIL import Image
topilimage = transforms.ToPILImage()
from torch.autograd import Variable
import numpy as np
# 데이터 핸들링
import nibabel as nib
from glob import glob
import itertools
import random
import os

# 시각화 및 진행률 표시
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from cyclegan_3d.base_model import BaseModel
from cyclegan_3d.cycle_gan_model import CycleGANModel
from cyclegan_3d.networks3D import define_G, define_D
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
# 디바이스 설정
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")



params = {
    # ✅ 데이터 설정
    'data_path': '../../data/registration_data/',     # Train images path
    'val_path': '../../data/registration_data',       # Validation images path
    'img_form': 'nii.gz',                                    # 이미지 포맷
    'batch_size': 1,
    'patch_size': [64,128, 128],                            # 3D 패치 크기 (H, W, D)
    'input_nc': 1,                                           # 입력 채널 수
    'output_nc': 1,                                          # 출력 채널 수
    'resample': False,
    'new_resolution': (0.45, 0.45, 0.45),
    'min_pixel': 0.1,
    'drop_ratio': 0,

    # ✅ 모델 구조 설정
    'ngf': 64,
    'ndf': 64,
    'netG': 'resnet_6blocks',
    'netD': 'n_layers',
    'n_layers_D': 3,
    'norm': 'instance',
    'no_dropout': True,

    # ✅ 학습 관련
    'isTrain': True,
    'model': 'cycle_gan',
    'direction': 'AtoB',
    'which_direction': 'AtoB',
    'phase': 'train',
    'gpu_ids': [5],
    'device': torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
    'workers': 4,

    'niter': 1000,                  # 학습 유지 epoch 수
    'niter_decay': 100,            # 학습률 감소 epoch 수
    'epoch_count': 1,
    'which_epoch': 'latest',
    'continue_train': False,
    'lr': 2e-4,
    'beta1': 0.5,
    'lr_policy': 'lambda',
    'lr_decay_iters': 50,
    'no_lsgan': False,
    'pool_size': 50,
    'lambda_A': 10.0,
    'lambda_B': 10.0,
    'lambda_identity': 0.5,

    # ✅ 초기화
    'init_type': 'normal',
    'init_gain': 0.02,

    # ✅ 저장/출력
    'checkpoints_dir': '../../model/translation/',
    'name': '3D_MRI_CT',
    'print_freq': 100,
    'save_latest_freq': 1000,
    'save_epoch_freq':1,
    'no_html': True,
    'verbose': True,
    'suffix': ''
}

class Preloaded3DDataset(Dataset):
    def __init__(self, data_a_list, data_b_list, input_size=(64,128, 128)):
        super().__init__()
        self.input_size = input_size  # (W, H, D)
        self.data_A = []
        self.data_B = []

        print("🔄 Loading and resizing NIfTI volumes into memory...")

        for a_path, b_path in tqdm(zip(data_a_list, data_b_list), total=len(data_a_list)):
            # Load and normalize
            vol_A = nib.load(a_path).get_fdata()
            vol_B = nib.load(b_path).get_fdata()

            # To Tensor: (1, D, H, W)
            tensor_A = torch.from_numpy(vol_A).unsqueeze(0).float()-1.
            tensor_B = torch.from_numpy(vol_B).unsqueeze(0).float()-1.

            # Resize to (1, D, H, W) → (1, 64, 128, 128)
            tensor_A = F.interpolate(tensor_A.unsqueeze(0), size=input_size, mode='nearest').squeeze(0)
            tensor_B = F.interpolate(tensor_B.unsqueeze(0), size=input_size, mode='nearest').squeeze(0)

            self.data_A.append(tensor_A)
            self.data_B.append(tensor_B)

        print(f"✅ Loaded {len(self.data_A)} volumes.")

    def __getitem__(self, index):
        # rand_index = random.randint(0, len(self.data_A) - 1)
        return {
            'A': self.data_A[index],
            'B': self.data_B[index]
        }

    def __len__(self):
        return len(self.data_A)
    
data_a_list = sorted(glob(os.path.join(params['data_path'], 'DWI/', '*.' + params['img_form'])))
data_b_list = [f.replace('DWI/', 'CT/') for f in data_a_list]
train_dataset = Preloaded3DDataset(data_a_list, data_b_list, input_size=(params['patch_size']))

# DataLoader 설정
train_loader = DataLoader(
    train_dataset,
    batch_size=params['batch_size'],
    shuffle=not params.get('serial_batches', False),
    num_workers=params.get('num_threads', 4)
)

model = CycleGANModel()

# 2. 초기화 (원래 argparse.Namespace를 받도록 되어 있음 → dict를 SimpleNamespace로 변환)
from types import SimpleNamespace
opt = SimpleNamespace(**params)
# 3. 모델 구조 초기화
opt.which_direction = opt.direction
opt.lambda_co_A = 2.0
opt.lambda_co_B = 2.0
model.initialize(opt)
model.setup(opt)
model.device = device
# model.load_networks(290)
from torchinfo import summary
# Generator A → B
print("🧠 Generator A → B (netG_A)")
summary(model.netG_A, input_size=(1, 1, params['patch_size'][0], params['patch_size'][1], params['patch_size'][1]), device=opt.device)



# 손실 함수
criterionGAN = nn.MSELoss()
criterionCycle = nn.L1Loss()
criterionIdt = nn.L1Loss()

# 이미지 풀 클래스 정의
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)
def save_volume_sample(real_A, real_B, model, epoch, save_dir="../../result/translation/3D_MRI2CT"):
    os.makedirs(save_dir, exist_ok=True)
    model.netG_A.eval()
    model.netG_B.eval()

    with torch.no_grad():
        fake_B = model.netG_A(real_A[:1])
        recon_A = model.netG_B(fake_B)

        fake_A = model.netG_B(real_B[:1])
        recon_B = model.netG_A(fake_A)

        def save_nii(tensor, name):
            np_img = tensor.squeeze().detach().cpu().numpy()
            nib.save(nib.Nifti1Image(np_img, affine=np.eye(4)), os.path.join(save_dir, f"{name}_epoch{epoch}.nii.gz"))

        save_nii(real_A[0,0], f"real_A")
        save_nii(fake_B[0], f"fake_B")
        save_nii(recon_A[0], f"recon_A")
        save_nii(real_B[0,0], f"real_B")
        save_nii(fake_A[0], f"fake_A")
        save_nii(recon_B[0], f"recon_B")
            
# 이미지 풀 초기화
fake_A_pool = ImagePool(opt.pool_size)
fake_B_pool = ImagePool(opt.pool_size)

# 손실 기록 리스트 초기화

# 에폭 반복
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # 데이터 로더에서 배치 반복
    D_losses = []

    G_losses = []
    cycle_losses = []
    idt_losses = []
    with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
        model.netG_A.train()
        model.netG_B.train()
        for data in tqdmDataLoader:
            
            real_A = data['A'].to(opt.device)
            real_B = data['B'].to(opt.device)

            # 모델에 입력 설정
            model.set_input([real_A, real_B]) 
            # 파라미터 최적화
            model.optimize_parameters()

            # 손실 값 기록

            D_losses.append((model.loss_D_B+model.loss_D_A).item())
            G_losses.append((model.loss_G_A + model.loss_G_B).item())
            cycle_losses.append((model.loss_cycle_A + model.loss_cycle_B).item())
            idt_losses.append((model.loss_idt_A + model.loss_idt_B).item())
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": f'Epoch {epoch}/{opt.niter + opt.niter_decay}',
                    "D Loss: ": f'{np.mean(D_losses):.4f}',
                    "G Loss: ":f'{np.mean(G_losses):.4f}',
                    "G-D Loss: ":f'{(np.mean(G_losses)-np.mean(D_losses)):.4f}',
                    "Cycle Loss": f'{np.mean(cycle_losses):.4f}',
                    "Idt Loss": f'{np.mean(idt_losses):.4f}',
                }
            )
    save_volume_sample(real_A, real_B, model, epoch)
    model.save_networks(epoch)

    # 학습률 업데이트
    model.update_learning_rate()
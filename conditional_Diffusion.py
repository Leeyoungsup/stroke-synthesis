import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import random
from conditionDiffusion.unet import Unet
from conditionDiffusion.embedding import ConditionalEmbedding
from conditionDiffusion.utils import get_named_beta_schedule
from conditionDiffusion.diffusion import GaussianDiffusion
from conditionDiffusion.Scheduler import GradualWarmupScheduler
from PIL import Image
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda",0)
print(f"Device:\t\t{device}")
import pytorch_model_summary as tms

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class_list = ['Normal', 'Ischemic','Hemorrhagic']
params = {'image_size': 512,
          'lr': 2e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 8,
          'epochs': 10000,
          'n_classes': None,
          'data_path': '../../data/2D_CT/',
          'image_count': 5000,
          'inch': 1,
          'modch': 64,
          'outch': 1,
          'chmul': [1, 2, 4, 8],
          'numres': 2,
          'dtype': torch.float32,
          'cdim': 256,
          'useconv': False,
          'droprate': 0.1,
          'T': 1000,
          'w': 1.8,
          'v': 0.3,
          'multiplier': 1,
          'threshold': 0.02,
          'ddim': True,
          }

trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5

class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self,parmas, images,label):
        
        self.images = images
        self.args=parmas
        self.label=label
        
    def trans(self,image):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)
            
        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)
            
        return image
    
    def __getitem__(self, index):
        image=self.images[index]
        label=self.label[index]
        image = self.trans(image)
        return image,label
    
    def __len__(self):
        return len(self.images)


image_label=[]
image_path=[]
for i in tqdm(range(len(class_list))):
    image_list=glob(params['data_path']+class_list[i]+'/*.png')
    if len(image_list)>params['image_count']:
        image_list=image_list[:params['image_count']]
    for j in range(len(image_list)):
        image_path.append(image_list[j])
        image_label.append(i)
        
train_images=torch.zeros((len(image_path),params['inch'],params['image_size'],params['image_size']))
for i in tqdm(range(len(image_path))):
    train_images[i]=trans(Image.open(image_path[i]).convert('L').resize((params['image_size'],params['image_size'])))
train_dataset=CustomDataset(params,train_images,image_label)
dataloader=DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True)

net = Unet(in_ch = params['inch'],
            mod_ch = params['modch'],
            out_ch = params['outch'],
            ch_mul = params['chmul'],
            num_res_blocks = params['numres'],
            cdim = params['cdim'],
            use_conv = params['useconv'],
            droprate = params['droprate'],
            dtype = params['dtype']
            ).to(device)
cemblayer = ConditionalEmbedding(len(class_list), params['cdim'], params['cdim']).to(device)
betas = get_named_beta_schedule(num_diffusion_timesteps = params['T'])
diffusion = GaussianDiffusion(
                    dtype = params['dtype'],
                    model = net,
                    betas = betas,
                    w = params['w'],
                    v = params['v'],
                    device = device
                )
optimizer = torch.optim.AdamW(
                itertools.chain(
                    diffusion.model.parameters(),
                    cemblayer.parameters()
                ),
                lr = params['lr'],
                weight_decay = 1e-4
            )

cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params['epochs']/100,
                            eta_min = 0,
                            last_epoch = -1
                        )
warmUpScheduler = GradualWarmupScheduler(
                        optimizer = optimizer,
                        multiplier = params['multiplier'],
                        warm_epoch = params['epochs'] // 10,
                        after_scheduler = cosineScheduler,
                        last_epoch = 0
                    )
# checkpoint=torch.load(f'../../model/conditionDiff/details/BRNT/ckpt_101_checkpoint.pt',map_location=device)
# diffusion.model.load_state_dict(checkpoint['net'])
# cemblayer.load_state_dict(checkpoint['cemblayer'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# warmUpScheduler.load_state_dict(checkpoint['scheduler'])
from torchinfo import summary

# 각 입력 데이터 생성
image_input = torch.randn(1, 1, 256, 256, device=device)  # 첫 번째 입력 (이미지 텐서)
random_int_input = torch.randint(1000, size=(1,), device=device)  # 두 번째 입력 (정수 텐서)
cemblayer_input = cemblayer(torch.Tensor([1]).long().to(device))  # 세 번째 입력 (cemblayer 텐서)

# 네트워크 요약 출력
summary(net, input_data=[image_input, random_int_input, cemblayer_input], col_names=["input_size", "output_size", "num_params"])


for epc in range(params['epochs']):
    diffusion.model.train()
    cemblayer.train()
    total_loss=0
    steps=0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            b = img.shape[0]
            optimizer.zero_grad()
            x_0 = img.to(device)
            lab = lab.to(device)
            cemb = cemblayer(lab)
            cemb[np.where(np.random.rand(b)<params['threshold'])] = 0
            loss = diffusion.trainloss(x_0, cemb = cemb)
            loss.backward()
            optimizer.step()
            steps+=1
            total_loss+=loss.item()
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss: ": total_loss/steps,
                    "batch per device: ":x_0.shape[0],
                    "img shape: ": x_0.shape[1:],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
            )
    warmUpScheduler.step()
    if (epc) % 5 == 0:
        diffusion.model.eval()
        cemblayer.eval()
        # generating samples
        # The model generate 80 pictures(8 per row) each time
        # pictures of same row belong to the same class
        all_samples = []
        each_device_batch =params['batch_size']//len(class_list)
        with torch.no_grad():

            lab = torch.ones(len(class_list), each_device_batch).type(torch.long) \
            * torch.arange(start = 0, end = len(class_list)).reshape(-1, 1)
            lab = lab.reshape(-1, 1).squeeze()
            lab = lab.to(device)
            cemb = cemblayer(lab)
            genshape = (each_device_batch*len(class_list) , params['outch'], params['image_size'], params['image_size'])
            if params['ddim']:
                generated = diffusion.ddim_sample(genshape, 50, 0, 'quadratic', cemb = cemb)
            else:
                generated = diffusion.sample(genshape, cemb = cemb)
            img = transback(generated)
            img = img.reshape(len(class_list), each_device_batch, params['outch'], params['image_size'], params['image_size']).contiguous()
            all_samples.append(img)
            samples = torch.concat(all_samples, dim = 1).reshape(len(class_list)*each_device_batch, params['outch'],params['image_size'], params['image_size'])
        create_dir(f'../../result/conditionDiff/CT/')

        save_image(samples,f'../../result/conditionDiff/CT/generated_{epc+1}_pict.png', nrow = each_device_batch)
        # save checkpoints
        checkpoint = {
                            'net':diffusion.model.state_dict(),
                            'cemblayer':cemblayer.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'scheduler':warmUpScheduler.state_dict()
                        }
        create_dir(f'../../model/conditionDiff/CT/')
        torch.save(checkpoint, f'../../model/conditionDiff/CT/ck1pt_{epc+1}_checkpoint.pt')
    torch.cuda.empty_cache()
    
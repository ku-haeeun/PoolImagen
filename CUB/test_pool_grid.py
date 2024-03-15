import torch
from imagen_pytorch.imagen_pytorch_pool import Unet, Imagen
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from tqdm import tqdm
import pickle
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import PIL
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 1 to use, 0 -> 1, 1 -> 2

#n_epochs = 10000
IMG_SIZE = 64 
IMG_DIR = '/home/haen/workspace/Imagen_org/CUB/cub-images/'
caption_file = '/home/haen/workspace/Imagen_org/CUB/cub_caption.csv'
SAVE_DIR ='/home/haen/workspace/Imagen_org/CUB/Cub_pool_32/'
CHECK_DIR = '/home/haen/workspace/Imagen_org/CUB/Cub_pool_32/checkpoint_pool/'
MODEL_DIR ='/home/haen/workspace/Imagen_org/CUB/Cub_pool_32/model(cub)_pool/'
LOG_DIR = '/home/haen/workspace/Imagen_org/CUB/Cub_pool_32/log_dir/train'

# Dataset
class CUB(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        
        with open(file='merge_caption_cub', mode='rb') as f:
            self.img_caption = pickle.load(f)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_caption[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

tf = transforms.Compose([
                               transforms.Resize(IMG_SIZE),       
                               transforms.CenterCrop(IMG_SIZE), 
                               transforms.ToTensor(),      
                               transforms.Normalize((0.5, 0.5, 0.5),  
                                                    (0.5, 0.5, 0.5)), 
                           ])

training_data = CUB(caption_file, IMG_DIR ,transform=tf)
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 2), # dimension multiplied
    num_resnet_blocks = (1, 2, 8), # each layer add resnet block
    layer_attns = (False, False, True), # whether to add attention for each layer
    layer_cross_attns = (False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = unet1,
    text_encoder_name = 'google/flan-t5-xxl',
    image_sizes = IMG_SIZE,
    timesteps = 100,
    cond_drop_prob = 0.1
).cuda()

#checkpoint path
checkpoint_path = MODEL_DIR
checkpoint_path = os.path.join(MODEL_DIR, 'model(cub)_poolcheckpoint_cub_pool_last.pth')
# state_dict load
checkpoint = torch.load(checkpoint_path)
imagen.load_state_dict(checkpoint['model_state_dict'])
        
# image path
output_test_dir_image = '/home/haen/workspace/Imagen_org/CUB/result_grid_pool_32/img/'
output_test_dir_text = '/home/haen/workspace/Imagen_org/CUB/result_grid_pool_32/text/'

# test
imagen.eval()

                
# 시드 값 설정
seed_value = 42

# PyTorch의 랜덤 시드 설정
torch.manual_seed(seed_value)

# CUDA 랜덤 시드 설정 (GPU 사용 시)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    
num_texts = len(training_data)
batch_size = 64

with torch.no_grad():
    for i in range(0, num_texts, batch_size):
        texts = training_data.img_labels.iloc[i:i+batch_size, 1].values.reshape((-1)).tolist()
        for text in texts:
            # 이미지 생성 시 랜덤 시드 설정
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
                
            images = imagen.sample(texts=[text] * 16, cond_scale=3.)
            to_pil = ToPILImage()
            grid = torch.cat([image.unsqueeze(0) for image in images], dim=0)
            pic = to_pil(make_grid(grid, nrow=4, padding=2, normalize=True, scale_each=True))
            pic.save(os.path.join(output_test_dir_image, f"{i}.jpg"))
            print(f"{i}")

            with open(os.path.join(output_test_dir_text, f"{i}.txt"), "w", encoding='utf-8') as f:
                f.write(str(text))
                
        
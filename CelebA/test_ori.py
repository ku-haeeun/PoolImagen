import torch
from imagen_pytorch.imagen_pytorch_pool import Unet, Imagen
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import pickle
import os
import pandas as pd
from torchvision.io import read_image
import PIL
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 1 to use, 0 -> 1, 1 -> 2

#n_epochs = 10000
IMG_SIZE = 64 
IMG_DIR = '/home/gail1/imagen_celeba/CelebAMask-HQ/CelebA-HQ-img/'
caption_file = '/home/gail1/imagen_celeba/CelebA/data/celeba_caption2.csv'
SAVE_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/output_pool/'
CHECK_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/checkpoint_pool/'
LOG_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/log_dir/'
MODEL_DIR ='/home/gail1/imagen_celeba/CelebA/celeba_pool_16/model(celeba)_16/'


# Dataset
class CUB(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        
        with open(file='merge_caption_celeba', mode='rb') as f:
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
    dim = 16,
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
checkpoint_path = os.path.join(MODEL_DIR, 'model(celeba)_16checkpoint_celeba_pool_reload.pth')
# state_dict load
checkpoint = torch.load(checkpoint_path)
imagen.load_state_dict(checkpoint['model_state_dict'])

#image path
output_test_dir_image = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/test_output_pool'
output_test_dir_text = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/test_output_text'
#test
imagen.eval()

num_texts = len(training_data) 
batch_size = 100  

with torch.no_grad():
    for i in range(0, num_texts, batch_size):
        
        images = imagen.sample(texts=training_data.img_labels.iloc[i:i+batch_size, 1].values.reshape((-1)).tolist(), cond_scale=3.)

        to_pil = ToPILImage()

        for j, (image, text) in enumerate(zip(images, training_data.img_labels.iloc[i:i+batch_size, 1].values)):

            img = to_pil(image.squeeze().cpu())
            img.save(os.path.join(output_test_dir_image, f"{i+j}.jpg"))
            print(i+j)
            
            with open(os.path.join(output_test_dir_text, f"{i+j}.txt"), "w", encoding='utf-8') as f:
                f.write(str(text)) 





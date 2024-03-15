import numpy as np
import os
import torch, torchvision, torchsummaryX
import torch.nn as nn, torch.optim as optim
import torchvision.datasets as dsets, torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.io import read_image
from torch.nn import functional as F
import torchvision.transforms as T
import pickle
from scipy.stats import entropy
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

# For Device

CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda:0' if CUDA else 'cpu')

inception = torchvision.models.inception_v3(pretrained= True).to(DEVICE)
inception.eval()    
print(inception)

#modify inception net's classifier for embedded vector
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
inception.fc=Identity()    
print("------------inception net after-------------------")
print(inception)


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

IMG_SIZE = 299
BATCH_SIZE = 10
IMG_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_pool_16/test_output_pool'
SAVE_PATH = 'img_pool_16.p'

# Dataset
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_list = [img_name for img_name in os.listdir(IMG_DIR) if img_name.endswith('.jpg')]
data_loader = DataLoader(img_list, batch_size=BATCH_SIZE, shuffle=False)

embedding_vectors = []



with torch.no_grad():
    for batch_img_names in tqdm(data_loader):
        batch_images = [preprocess_image(os.path.join(IMG_DIR, img_name), transform) for img_name in batch_img_names]
        batch_images = torch.stack(batch_images).to(DEVICE)
        embeddings = inception(batch_images)
        embedding_vectors.append(embeddings.cpu().numpy())

embedding_vectors = np.concatenate(embedding_vectors, axis=0)

# Save the embedding vectors to a pickle file
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(embedding_vectors, f)

print("Embedding vectors saved to", SAVE_PATH)
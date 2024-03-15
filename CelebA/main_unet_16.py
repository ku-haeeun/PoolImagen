import torch
from imagen_pytorch import Unet, Imagen
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import pickle
import os
import pandas as pd
from torchvision.io import read_image
import PIL
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 1 to use, 0 -> 1, 1 -> 2

n_epochs = 7000
IMG_SIZE = 64 
IMG_DIR = '/home/gail1/imagen_celeba/CelebAMask-HQ/CelebA-HQ-img/'
caption_file = '/home/gail1/imagen_celeba/CelebA/data/celeba_caption2.csv'
SAVE_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_ori_16/output(celeba)_16/'
CHECK_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_ori_16/checkpoint(celeba)_16/'
LOG_DIR = '/home/gail1/imagen_celeba/CelebA/celeba_ori_16/log_dir/'
MODEL_DIR ='/home/gail1/imagen_celeba/CelebA/celeba_ori_16/model(celeba)_16'


# Dataset
class CelebAHQ(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        
        with open(file='merge_caption_celeba', mode='rb') as f:
            self.img_caption = pickle.load(f)
            # print(pickle.load(f))

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
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

training_data = CelebAHQ(caption_file, IMG_DIR ,transform=tf)
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

#optimizer_1 = torch.optim.Adam(unet1.parameters(), lr=0.00001)
optimizer_2 = torch.optim.Adam(imagen.parameters(), lr=0.0001)
optimizer = [optimizer_2]

# feed images into imagen, training each unet in the cascade

texts = [
              'The person has high cheekbones, and pointy nose. She is wearing lipstick',
              'This person has arched eyebrows, wavy hair, and mouth slightly open. She wears lipstick. She is attractive.',
              'She wears lipstick. She is smiling, and attractive and has wavy hair, and brown hair.',
              'He has big lips, wavy hair, and big nose.',
              'This man has bags under eyes, double chin, rosy cheeks, and big nose and wears hat. He is smiling.'
            ]
text_embeds = torch.stack(training_data.img_caption[:5]).cuda()
print(text_embeds.shape)

transform_img = transforms.ToPILImage()

summary = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'train_celeba_16'))

for idx_unet in [1]:
    for epoch in range(n_epochs):
        images = imagen.sample(text_embeds = text_embeds, cond_scale = 3.)
        for i in range(images.shape[0]):
            img = transform_img(images[i,:,:,:].squeeze()) # 축소
            img.save(os.path.join(SAVE_DIR) + "{}.jpg".format(i))
        train_dataloader_tqdm = tqdm(train_dataloader)
        
        for i, (inputs, targets) in enumerate(train_dataloader_tqdm):
            loss = imagen(inputs.cuda(), text_embeds = targets.cuda(), unet_number = idx_unet)
            loss.backward()
            optimizer[idx_unet-1].step()
            optimizer[idx_unet-1].zero_grad()
            train_dataloader_tqdm.set_description("Loss %.04f | epoch %d | step %d" % (loss.item(), epoch, i))
            
        if epoch % 10 == 0: # 매 10 epoch마다
            summary.add_scalar('Loss', loss.item(), epoch)
            
        if epoch % 10 == 9 or epoch == 0:
            torch.save({
              'epoch': epoch,
              'model_state_dict': imagen.state_dict(),
              'optimizer_state_dict': optimizer[idx_unet-1].state_dict(),
              'loss': loss.item(),
            }, 'checkpoint_celeba_16.pth')
            torch.save(imagen.state_dict(), '{0}/epoch_{1}'.format(CHECK_DIR, epoch))
            
summary.close()
        

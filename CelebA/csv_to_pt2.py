import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import os
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 2 to use

f = pd.read_csv('/home/gail1/imagen_celeba/CelebA/data/celeba_caption2.csv', header=None)
print(f.shape)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5EncoderModel.from_pretrained("google/flan-t5-xxl")

res = []

for i in range(20000,30000):
    input_ids = tokenizer.encode(f.iloc[i, 1], return_tensors="pt", padding='max_length', max_length=256)
    outputs = model(input_ids=input_ids)
    outputs = outputs[0].squeeze()
    res.append(outputs.detach())
    print(i)


with open("xl_t5_caption_celeba_3", "wb") as fp:   #Pickling
    pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    

import pickle
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"  # Set the GPU 2 to use

file_list = ['xl_t5_caption_celeba_1','xl_t5_caption_celeba_2' ,'xl_t5_caption_celeba_3']

# 데이터를 저장할 빈 리스트 생성
data = []

# 파일 하나씩 열어서 데이터 추가
for file in file_list:
    with open(file, 'rb') as f:
        data += pickle.load(f)

# 병합된 데이터 확인
with open('merge_caption_celeba', 'wb') as f:
    pickle.dump(data, f,protocol=pickle.HIGHEST_PROTOCOL)

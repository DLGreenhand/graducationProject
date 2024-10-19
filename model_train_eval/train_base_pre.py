import requests,os,time
from PIL import Image
import json, sys, random
from transformers import AutoModelForSeq2SeqLM, Pix2StructProcessor,Pix2StructForConditionalGeneration
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
import torch,shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocess import preprocess_fn, get_cpn_info
    
save_path = 's2w_wid_pre_func' # 保存模型路径文件夹
fn = [
    'resize',
    # 'caption',
    # 'gray'
]
model_path = 'screen2words' # load预训练模型路径文件夹
device = "cuda:5"
learning_rate = 1e-6
weight_decay = 0

# 获得train集的所有图片id

# 加载pix2struct-base预训练模型
model = Pix2StructForConditionalGeneration.from_pretrained(f"../../models/{model_path}").to(device)
processor = Pix2StructProcessor.from_pretrained(f"../../models/{model_path}")
processor.image_processor.is_vqa = True
text_processor = Pix2StructProcessor.from_pretrained(f"../../models/{model_path}")
text_processor.image_processor.is_vqa = False

screen2words_ids = set(pd.read_csv('../../dataset/screen2words/screen_summaries.csv')['screenId'].unique())
start_time=time.time()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

all_set = set()
for file in os.listdir(f"../../dataset/rico/combined/"):
    name,ext = os.path.splitext(file)
    all_set.add(int(name))

all_set = set(all_set.difference(screen2words_ids))
all_set = set(random.sample(list(all_set),int(len(all_set)*0.5)))
train_set = random.sample(list(all_set),int(len(all_set)*0.8))
eval_set = all_set.difference(train_set)
epoch = 100

# if os.path.exists(f"../../models/{save_path}/loss.txt"):
#     os.remove(f"../../models/{save_path}/loss.txt")

max_cider = 0
from calCIDEr import Cider
cider_cal = Cider()

train_set = list(train_set)
eval_set = list(eval_set)

for i in range(epoch):
    loss_all = 0.0
    model.train()
    if not os.path.exists(f"../../models/{save_path}_epoch_{i}"):
        os.mkdir(f"../../models/{save_path}_epoch_{i}")
    for idx in tqdm(train_set):
        url = f"../../dataset/rico/combined/{idx}.jpg"
        image = Image.open(url)
        image = preprocess_fn(image,fn,idx)
        cpns = get_cpn_info(idx)
        sample_loss = 0.0
        if len(cpns) ==0: continue
        for cpn in cpns:
            inputs = processor(
                images=image, 
                return_tensors="pt",
                header_text=f"what function is it in the box {cpn[0]}?",
                font_path='./Arial.ttf', 
                # truncation=True,
                # padding="max_length", 
                # max_length=2048
            ).to(device)
            labels = text_processor(
                text=cpn[1], 
                return_tensors="pt", 
                # padding="max_length", 
                # max_length=2048
            ).input_ids.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss_all += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sample_loss+=loss
        with open(f"../../models/{save_path}_epoch_{i}/loss_tmp.txt",'a') as fp:
            fp.write(f"epoch:{i},loss:{sample_loss/len(cpns)}\n")
    
    model.save_pretrained(f"../../models/{save_path}_epoch_{i}")
    try:
        with open(f"../../models/{save_path}_epoch_{i}/loss.txt",'a') as fp:
            fp.write(f"epoch:{i},loss:{loss_all/len(train_set)}\n")
    except: print(f"fault!")        
    
print(time.time()-start_time)
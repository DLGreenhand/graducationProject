import requests,os,time
from PIL import Image
import torch, sys, random
from transformers import Pix2StructVisionModel, Pix2StructProcessor,Pix2StructForConditionalGeneration
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

save_path = 'base_ft_rand_resize'
is_rand = True
longest = False
resize = True
model_path = 'base'
device = "cuda:6"

# 获得train集的所有图片id
with open('../../dataset/screen2words/split/train_screens.txt','r') as fp:
    s = fp.read()
    train_set = s.split('\n')
train_set.pop()
train_set=set(train_set)

# 加载pix2struct-base预训练模型
model = Pix2StructForConditionalGeneration.from_pretrained(f"../../models/{model_path}").to(device)
processor = Pix2StructProcessor.from_pretrained(f"../../models/screen2words")
processor.image_processor.is_vqa = False
# print(model)
# exit()
# model= nn.DataParallel(model,device_ids = [1,2])

# 获取所有图片id对应的摘要list数据集（长度为5）
summary_dict = dict()
screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()
start_time=time.time()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

epoch = 100

for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    summaries = row['summary']
    summary_dict[idx] = summaries
    if idx not in train_set: continue
        # print("fucking"+idx)
        
train_set = list(train_set)
print(len(train_set))
min_loss = sys.maxsize
for i in range(epoch):
    loss_all = 0.0
    for idx in tqdm(train_set):
        url = f"../../dataset/rico/combined/{idx}.jpg"
        summaries = summary_dict[idx]
        image = Image.open(url)
        if resize and image._size==(1080,1920):
            image = image.resize((544,960))
        refs = []
        if longest:
            text = ""
            for summary in summaries:
                if len(summary)>len(text):
                    text=summary
            refs.append(text)
        elif is_rand:
            refs.append(random.choice(summaries))
        else:
            refs = summaries
        for summary in refs:
            inputs = processor(
                images=image, 
                return_tensors="pt",
                # font_path='./Arial.ttf', 
                # truncation=True,
                # padding="max_length", 
                # max_length=2048
            ).to(device)
            labels = processor(
                text=summary, 
                return_tensors="pt", 
                # truncation=True,
                # padding="max_length", 
                # max_length=2048
            ).input_ids.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
    print(f"epoch:{i},loss:{loss_all/len(train_set)}")
    if loss_all < min_loss:
        min_loss = loss_all
        model.save_pretrained(f"../../models/{save_path}")
    try:
        with open(f"../../models/{save_path}/loss.txt",'a') as fp:
            fp.write(f"epoch:{i},loss:{loss_all/len(train_set)}")
    except: print(f"fault! loss:{loss_all/len(train_set)}")        
print(time.time()-start_time)
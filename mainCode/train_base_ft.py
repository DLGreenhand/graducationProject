import requests,os,time
from PIL import Image
import json, sys, random
from transformers import Pix2StructVisionModel, Pix2StructProcessor,Pix2StructForConditionalGeneration
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
import torch,shutil
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocess import preprocess_fn
from get_data import get_s2w_data
    
save_path = 's2w_ft_caption' # 保存模型路径文件夹
rand_each = not True # 每轮都随机选caption
fn = [
    'resize',
    'caption',
    # 'gray'
]
model_path = 'screen2words' # load预训练模型路径文件夹
device = "cuda:7"
learning_rate = 1e-5
weight_decay = 0

if not os.path.exists(f"../../models/{save_path}/"):
    os.mkdir(f"../../models/{save_path}/")

# 获得train集的所有图片id
train_set = get_s2w_data("train")
test_set = get_s2w_data("test")

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
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

epoch = 100
# 记录5条参考文句
gts_dict = {}
train_dict = {}

if os.path.exists(f"../../models/{save_path}/loss.txt"):
    os.remove(f"../../models/{save_path}/loss.txt")

for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    # if idx not in train_set and idx not in test_set: continue
    summaries = row['summary']
    if idx in test_set: gts_dict[idx] = summaries
    else :train_dict[idx] = summaries
    summary_dict[idx] = random.choice(summaries)
  
train_set = list(train_set)
test_set = list(test_set)
max_cider = 0
from calCIDEr import Cider
cider_cal = Cider()

for i in range(epoch):
    loss_all = 0.0
    model.train()
    for idx in tqdm(train_set):
        url = f"../../dataset/rico/combined/{idx}.jpg"
        image = Image.open(url)
        image = preprocess_fn(image,fn,idx)
        if rand_each:
            summary = random.choice(train_dict[idx])
        else :
            summary = summary_dict[idx]
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
    
    # 选cider最高的模型保存
    model.eval()
    with torch.no_grad():
        res_dict = {}
        res = []
        for idx in tqdm(test_set):
            url = f"../../dataset/rico/combined/{idx}.jpg"
            image = Image.open(url)
            image = preprocess_fn(image,fn,idx)
            inputs = processor(
                images=image, 
                return_tensors="pt",
                font_path='./Arial.ttf', 
                # truncation=True,
                # padding="max_length", 
                # max_length=2048
            ).to(device)
            prediction = model.generate(**inputs)
            caption = processor.decode(prediction[0], skip_special_tokens=True)
            res_dict[idx] = [caption]
            res.append(caption)

        cider = cider_cal.compute_score(gts_dict, res_dict)
        print(f"epoch:{i} end")
        record = {
            "scores":cider[1].tolist(),
            "caption_res":res
        }
    if max_cider < cider[0]:
        max_cider = cider[0]
        model.save_pretrained(f"../../models/{save_path}")
    try:
        with open(f"../../models/{save_path}/loss.txt",'a') as fp:
            fp.write(f"epoch:{i},loss:{loss_all/len(train_set)},cider:{cider[0]}\n")
    except: print(f"fault! loss:{loss_all/len(train_set)}")        
    with open(f"../../models/{save_path}/{i}.json",'w') as fp:
        json.dump(record,fp)
print(time.time()-start_time)
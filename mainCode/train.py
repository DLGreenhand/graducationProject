import requests,os,time
from PIL import Image
import torch
from transformers import Pix2StructVisionModel, AutoProcessor,Pix2StructForConditionalGeneration
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self):
        self.data = []

    def add_data(self, new_data):
        # image = new_data[0]
        # text=new_data[1]
        # image = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((960, 512)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])(image)
        self.data.append(new_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][0]
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((960, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
        text = self.data[index][1]
        return image, text

# 获得train集的所有图片id
with open('../../dataset/screen2words/split/train_screens.txt','r') as fp:
    s = fp.read()
    train_set = s.split('\n')
train_set.pop()
train_set=set(train_set)

# 加载pix2struct-base预训练模型
device = "cpu"
model = Pix2StructForConditionalGeneration.from_pretrained("../../models/screen2words").to(device)
processor = AutoProcessor.from_pretrained("../../models/screen2words")
processor.image_processor.is_vqa = False
# print(model)
# exit()
# model= nn.DataParallel(model,device_ids = [2,3,4,7])

# 获取所有图片id对应的摘要list数据集（长度为5）
summary_dict = dict()
screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()
mydata = MyDataset()
screen2words=screen2words.head(100)
for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    summaries = row['summary']
    summary_dict[idx] = summaries
    if idx not in train_set: continue
    # print("fucking"+idx)
    url = f"../../dataset/rico/combined/{idx}.jpg"
    image = Image.open(url)
    for summary in summary_dict[idx]:
        mydata.add_data([image,summary])
    # print(summaries)
print("dataset complete")
start_time=time.time()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
res = []

dataloader = DataLoader(mydata, num_workers=2, batch_size=2, shuffle=True)
epoch=1
for i in range(epoch):
    for image, text in tqdm(dataloader):
        inputs = processor(
            images=image, 
            return_tensors="pt",
            # font_path='./Arial.ttf', 
            # truncation=True,
            # padding="max_length", 
            # max_length=2048
        ).to(device)
        print
        labels = processor(
            text=summary, 
            return_tensors="pt", 
            # truncation=True,
            # padding="max_length", 
            # max_length=2048
        ).input_ids.to(device)
        outputs = model(**inputs, labels=labels)
        # print((outputs.last_hidden_state.shape))
        # loss = outputs.loss
        # loss.backward()
        # optimizer.step()
        
        # predictions = model.generate(**inputs)
        # summary_dict[idx].append(processor.decode(predictions[0], skip_special_tokens=True))
        # for s in screen_summary[idx]:
        #     res.append([idx,s])
print(time.time()-start_time)

# df = pd.DataFrame(res,columns=['id','res'])
# df.to_csv('results.csv')
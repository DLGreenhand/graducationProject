import sys,time
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 几个运行参数
num_workers = 2
batch_size = 2
save_path = 's2w_ft_resize'
is_rand = True
longest = False
resize = True
model_path = 'screen2words'
device = "cuda:6"

# 获得train集的所有图片id
with open('../../dataset/screen2words/split/train_screens.txt','r') as fp:
    s = fp.read()
    train_set = s.split('\n')
train_set.pop()
train_set=set(train_set)

# 加载pix2struct-base预训练模型
model = Pix2StructForConditionalGeneration.from_pretrained(f"../../models/{model_path}").to(device)
processor = AutoProcessor.from_pretrained("../../models/screen2words")
processor.image_processor.is_vqa = False
# print(model)
# exit()
model= nn.DataParallel(model,device_ids = [6,7])

# 构造pytorch数据集加载器
class MyDataset(Dataset):
    def __init__(self, processor):
        self.dataset = []
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def add_data(self,data):
        self.dataset.append({
            "image":data[0],
            "text":data[1],
            # "id":data[2]
        })
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if resize:
            image = item["image"].resize((544,960))
        else:
            image = item["image"]
        encoding = self.processor(images=image, return_tensors="pt", add_special_tokens=True)
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item["text"]
        # encoding["id"] = item["id"]
        return encoding

def collator(batch):
    new_batch = {"flattened_patches":[], "attention_mask":[]}
    texts = [item["text"] for item in batch]
    # ids = [item["id"] for item in batch]
    text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=20)
    
    new_batch["labels"] = text_inputs.input_ids
    # new_batch["ids"] = ids
    for item in batch:
        new_batch["flattened_patches"].append(item["flattened_patches"])
        new_batch["attention_mask"].append(item["attention_mask"])
    
    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

    return new_batch

# 获取所有图片id对应的摘要list数据集（长度为5）
summary_dict = dict()
screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()
mydata = MyDataset(processor)
# screen2words=screen2words.head(100)
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
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
res = []

dataloader = DataLoader(mydata, num_workers=num_workers, batch_size=batch_size, shuffle=True, collate_fn=collator)
epoch=100

# 训练模型
model.train()
min_loss = sys.maxsize
for i in range(epoch):
    loss_all = 0.0
    for idx, batch in tqdm(enumerate(dataloader)):
        # print(batch.pop("ids"))
        labels = batch.pop("labels").to(device)
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        outputs = model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        # print(loss)
        loss_all += loss.sum()
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"epoch:{i},loss:{loss_all/len(train_set)}")
    if loss_all < min_loss:
        min_loss = loss_all
        model.save_pretrained(f"../../models/{save_path}")
    try:
        with open(f"../../models/{save_path}/loss.txt",'a') as fp:
            fp.write(f"epoch:{i},loss:{loss_all/len(train_set)}\n")
    except: print(f"fault! loss:{loss_all/len(train_set)}") 
        
print(time.time()-start_time)

# df = pd.DataFrame(res,columns=['id','res'])
# df.to_csv('results.csv')
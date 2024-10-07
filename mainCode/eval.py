"""
模型推理和验证模块：
    生成screen对应caption, 和5个标注caption计算CIDEr指标
"""

import time,json
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, Pix2StructProcessor
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from calCIDEr import Cider
from preprocess import preprocess_fn

eval_model = "base_ft_caption"
device = "cuda:7"
fn = [
    'resize',
    'caption',
    # 'gray'
]

# 获得train集的所有图片id
with open('../../dataset/screen2words/split/test_screens.txt','r') as fp:
    s = fp.read()
    eval_set = s.split('\n')
eval_set.pop()
eval_set=set(eval_set)

# 加载pix2struct-base预训练模型
model = Pix2StructForConditionalGeneration.from_pretrained(f"../../models/{eval_model}").to(device)
processor = Pix2StructProcessor.from_pretrained("../../models/screen2words")
processor.image_processor.is_vqa = False

# 获取所有图片id对应的摘要list数据集（长度为5）
gts_dict = dict()
res_dict = dict()
screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()

cider = Cider()
sample_num = 0
res = []
start_time=time.time()

for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    summaries = row['summary']
    if idx not in eval_set: continue
    sample_num += 1
    gts_dict[idx] = summaries

    # url = 'image.jpg'
    url = f"../../dataset/rico/combined/{idx}.jpg"
    image = Image.open(url)
    image = preprocess_fn(image,fn,idx)
    inputs = processor(
        images=image, 
        return_tensors="pt",
        font_path = './Arial.ttf',
        # padding="max_length", 
        # max_length=2048
        ).to(device)
    prediction = model.generate(**inputs)
    caption = processor.decode(prediction[0], skip_special_tokens=True)
    res_dict[idx] = [caption]

    res.append([idx, summaries, caption])

df = pd.DataFrame(res,columns=['id','summaries','caption'])
df.to_csv('test_results.csv')

print(f"样本数:{sample_num},CIDEr:{cider.compute_score(gts_dict,res_dict)}")
print(f"推理耗时:{time.time()-start_time}")
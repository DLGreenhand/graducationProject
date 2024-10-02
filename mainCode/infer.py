"""
模型推理和验证模块：
    生成screen对应caption, 和5个标注caption计算CIDEr指标
"""

import time
from PIL import Image
import torch, requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, Pix2StructProcessor
import pandas as pd
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from calCIDEr import cider
import numpy as np

# 获得train集的所有图片id
with open('../../dataset/screen2words/split/test_screens.txt','r') as fp:
    s = fp.read()
    eval_set = s.split('\n')
eval_set.pop()
eval_set=set(eval_set)

# 加载pix2struct-base预训练模型
device = "cuda:7"
model = Pix2StructForConditionalGeneration.from_pretrained("../../models/screen2words").to(device)
processor = Pix2StructProcessor.from_pretrained("../../models/screen2words")
processor.image_processor.is_vqa = False

start_time=time.time()

url = 'image.jpg'
# url = f"../../dataset/rico/combined/{idx}.jpg"
image = Image.open(url)
image = image.convert("L")
image.save("gray.jpg")
inputs = processor(
        images=image, 
        return_tensors="pt",
        font_path = './Arial.ttf',
        # padding="max_length", 
        # max_length=2048
        ).to(device)
prediction = model.generate(**inputs)
caption = processor.decode(prediction[0], skip_special_tokens=True)
print(caption)
print(time.time()-start_time)
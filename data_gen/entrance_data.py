import os
from PIL import Image
import json

with open('../../dataset/screen2words/split/train_screens.txt','r') as fp:
    s = fp.read()
    sample_set = s.split('\n')
sample_set.pop()
sample_set=set(sample_set)

for idx in sample_set:
    url = f"../../dataset/rico/combined/{idx}.json"
    print(url)
    with open(url) as fp:
        d = json.load(fp)
        print(d)
        break
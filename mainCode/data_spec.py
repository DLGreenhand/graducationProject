from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil,json

def search(tree:dict,res_dict:dict):
    if "text" in tree.keys() and "bounds" in tree.keys():
        res_dict['']

screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()
st = set()
cnt=0
for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    if idx in st: continue
    st.add(idx)
    # url = f"../../dataset/rico/combined/{idx}.jpg"
    # image = Image.open(url)
    try:
        with open(f"../../dataset/rico/combined/{idx}.json") as fp:
            # tree = json.load(fp)['activity']['root']
            # res_dict = {"h":0,"cnt":0}
            # search(tree,res_dict)
            act_name = json.load(fp)['activity_name']
            if '/' in act_name:
                act_name=act_name.split('/')[-1]
            cnt = max(cnt,len(act_name))
    except: continue
print(cnt)
        
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil,json,os,string

screen2words_ids = set(pd.read_csv('../../dataset/screen2words/screen_summaries.csv')['screenId'].unique())
file_path = "../../dataset/rico/semantic"

cpn_names = set()

def check_characters(s):
    # 所有大小写字母
    letters = set(string.ascii_letters)
    # 字符串中的唯一字符
    unique_chars = set(s)
    # 求差集，找出除了大小写字母以外的字符
    other_chars = unique_chars - letters
    return other_chars
cnt=0
def search_cpn(root:dict,cpn_set:set):
    # if 'componentLabel' in root: cpn_set.add(root['componentLabel'])
    global cnt
    cnt+=1
    if 'resource-id' in root:
        char_st = check_characters(root['resource-id'].split("/")[-1])
        for c in char_st:
            cpn_set.add(c)
    if 'children' not in root: return
    for child in root['children']:
        search_cpn(child,cpn_set)

for file in os.listdir(file_path):
    name,ext=os.path.splitext(file)
    if 'json' not in ext or int(name) in screen2words_ids:
        continue
    with open(f"{file_path}/{name}.json") as fp:
        sem = json.load(fp)
    cpn_num = search_cpn(sem,cpn_names)
    if cpn_num==1:print(file)

print(cpn_names)
print(cnt/43844)

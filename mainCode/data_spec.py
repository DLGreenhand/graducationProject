from PIL import Image
import pandas as pd
from tqdm import tqdm

screen2words = pd.read_csv('../../dataset/screen2words/screen_summaries.csv').groupby('screenId')['summary'].agg(list).reset_index()
st = set()
cnt=0
for _, row in tqdm(screen2words.iterrows()):
    idx = str(row['screenId'])
    url = f"../../dataset/rico/combined/{idx}.jpg"
    image = Image.open(url)
    st.add((image._size))
    if image._size==(1080,1920):
        image.save('im.jpg')
        image = image.resize((544,960))
        image.save('resize.jpg')
        print(image._size)
        break
print(cnt/len(screen2words))
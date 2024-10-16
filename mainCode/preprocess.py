from PIL import Image, ImageDraw, ImageFont
import json

# 渲染caption策略
def preprocess_fn(image:Image.Image,fn:set,idx:str):
    """_summary_

    Args:
        image (Image.Image): PIL图像
        fn (set): 变换集合
        idx (str): 图片id
    """
    if 'resize' in fn:
        image = image.resize((540,960))
        if 'caption' in fn:
            image = add_title(image,50,idx)
    
    if 'gray' in fn:
        image = image.convert("L")
        
    return image


def add_title(image:Image.Image,add_height:int,idx:str):
    """_summary_

    Args:
        image (Image.Image): PIL图像
        title (str): 添加文字
        add_height (int): 顶部添加高度
        idx (str): 图片id

    Returns:
        _type_: 变换后的图片
    """
    
    title = ""
    try:
        with open(f"../../dataset/rico/combined/{idx}.json") as fp:
            title = json.load(fp)['activity_name'].split('/')[-1]
            if len(title)>24:title=title[-24:]
    except: 
        print(f"{idx}.json fault!")
        return image
    
    # 获取图片的尺寸
    width, height = image.size

    # 创建一个白色区域，高度为50像素，宽度与图片相同
    white_area = Image.new('RGB', (width, add_height), 'white')

    # 将白色区域与原图片合并
    new_image = Image.new('RGB', (width, height + add_height))
    new_image.paste(white_area, (0, 0))
    new_image.paste(image, (0, add_height))

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(new_image)

    # 准备要添加的文字
    
    # 选择字体和大小，如果系统中没有该字体，会使用默认字体
    try:
        font = ImageFont.truetype("Arial.ttf", int(0.9*add_height))
    except IOError:
        font = ImageFont.load_default()

    # 设置文字颜色
    text_color = (0, 0, 0)  # 黑色

    # 设置文字位置（白色区域的中央）
    text_x = (width - draw.textlength(title, font=font)) / 2
    text_y = 0  # 距离白色区域顶部10像素

    # 在白色区域上渲染文字
    draw.text((text_x, text_y), title, font=font, fill=text_color)
    return new_image

# 预训练策略
skip_char = ['$', 'в', '1', 'к', '?', 'ш', '3', '2', '0', '8', 'е', ':', '5', '9', '_', '6', '.', ' ', '7', 'р', '4']

def convert_xy(bounds:list):
    bounds[0] = int(bounds[0] * (540/1440))
    bounds[2] = int(bounds[2] * (540/1440))
    bounds[1] = int(bounds[1] * (960/2560))
    bounds[3] = int(bounds[3] * (960/2560))

def search_cpns(root:dict,cpns:list):
    if 'children' not in root:return
    if 'componentLabel' in root and 'bounds' in root:
        s = root['componentLabel']
        convert_xy(root["bounds"])
        if 'clickable' in root:
            if root['clickable']:
                s = f"{s}, clickable"
            else :s = f"{s}, not clickable"
        if 'textButtonClass' in root:
            s = f"{s}, {root['textButtonClass']}"
        if 'iconClass' in root:
            s = f"{s}, {root['iconClass']}"
        # if 'resource-id' in root:
        #     tmp = str(root['resource-id'].split("/")[-1])
        #     for c in skip_char:
        #         tmp=tmp.replace(c," ")
        #     s = f"{s}, {tmp}"
        cpns.append([root['bounds'],s])
    for child in root['children']:
        search_cpns(child,cpns)

def get_cpn_info(idx):
    file_path = "../../dataset/rico/semantic"
    cpns = []
    with open(f"{file_path}/{idx}.json") as fp:
        root = json.load(fp)
        if 'children' not in root: return cpns
    search_cpns(root,cpns)
    return cpns


if __name__ == "__main__":
    image = Image.open("image.jpg")
    image = preprocess_fn(image,['resize'],0)
    print(image.size)
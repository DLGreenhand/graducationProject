from PIL import Image, ImageDraw, ImageFont
import json

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

if __name__ == "__main__":
    image = Image.open("image.jpg")
    image = preprocess_fn(image,['resize'],0)
    print(image.size)
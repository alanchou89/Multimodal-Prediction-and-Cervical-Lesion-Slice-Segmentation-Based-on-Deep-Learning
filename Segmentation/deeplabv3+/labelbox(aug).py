# 使用.ipynb分段執行

###============================================================================###
# 解析.ndjson檔案
import json

#  NDJSON 文件路徑
file_path = r'NDJSON 文件路徑/.ndjson'
# Export v2 project - image_A(100) - 3_6_2024.ndjson
# Export v2 project - image_A(101~500) - 3_14_2024.ndjson

# 初始化字典来保存 mask 和 composite_mask 的 URL 以及相應的分類名稱和圖像名稱
class_mask_urls = []                # 儲存 "影像名稱"、"影像分類"、"url"
composite_mask_urls = []            # 儲存 "影像名稱"、"影像分類"、"url"
id_urls = []                        # 儲存影象名稱 ex: 0002A0、0005A0...
id_counts = []                      # 儲存影象名稱和次數 

# 讀取 NDJSON文件
try:
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            data = json.loads(line)

            # 提取圖像名稱
            id_url = data.get("data_row", {}).get("external_id", "").split('.')[0]
            id_urls.append(id_url)

            # 遍歷所有標註
            annotations = data.get("projects", {}).get(next(iter(data.get("projects", {}))), {}).get("labels", [])[0].get("annotations", {}).get("objects", [])
            id_count = len(annotations)
            id_counts.append((id_url, id_count))
            
            for obj in annotations:
                name = obj.get("name", "Unknown")
                mask_url = obj.get("mask", {}).get("url")
                composite_mask_url = obj.get("composite_mask", {}).get("url")
                
                # 在添加到 class_mask_urls 時，同時包含分類名稱、URL 和圖像名稱
                if mask_url:
                    class_mask_urls.append({"image_name": id_url, "name": name, "url": mask_url})
                
                if composite_mask_url:
                    composite_mask_urls.append({"image_name": id_url, "name": name, "url": composite_mask_url})

except Exception as e:
    print(f"發生錯誤：{e}")
###============================================================================###

###============================================================================###
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from pathlib import Path

#  Labelbox API 密鑰
api_key = "Labelbox API 密鑰"
headers = {
    "Authorization": f"Bearer {api_key}"
}

###################################################################################
# class_mask_urls陣列中提取影像名稱、分類、網址
# 根據網址下載影像(黑、白，白色代表label位置)，並根據分類更改顏色
# 如果 class_mask_urls的影像名稱有多個分類，則將其合併
# ex:若有2個分類分別為'AWE'、'Atypical'，則分別根據網址取出圖象並修改兩張影像之顏色，
# 在將兩張影像合併起來成為一組新影像再將其根據路徑儲存，名稱為該'image_name'名稱+.png
###################################################################################

# 颜色映射
color_map = {
    'AWE': (255, 255, 0),  # 黄色          # 255, 255, 0
    'Punctation': (255, 0, 0),  # 红色     # 255, 0, 0
    'Mosaic': (0, 255, 0),  # 绿色         # 0, 255, 0
    'Atypical': (0, 0, 255)  # 蓝色        # 0, 0, 255
}

# 下載並修改遮罩顏色
def download_and_modify_mask(url, classification):
    response = requests.get(url, headers=headers)
    mask_img = Image.open(BytesIO(response.content)).convert("RGBA")
    data = np.array(mask_img)
    r, g, b, a = data.T
    white_areas = (r == 255) & (b == 255) & (g == 255)
    data[..., :-1][white_areas.T] = color_map[classification]  # 更改颜色
    return data  # 返回修改后的 numpy 数组

# 合併圖像
def merge_images(images):
    # 所有圖像尺寸相同
    merged_data = np.sum(images, axis=0)  # 直接相加所有圖像的像素值
    np.clip(merged_data, 0, 255, out=merged_data)  # 確保像素值在合理範圍內
    return Image.fromarray(merged_data.astype(np.uint8))

# 保存路径
base_save_path = Path(r'遮罩路徑\mask')

last_image_name = None
np_images_to_merge = []

for item in class_mask_urls:
    image_name = item['image_name']
    classification_name = item['name']
    mask_url = item['url']
    
    # 下载並修改颜色
    modified_np_mask = download_and_modify_mask(mask_url, classification_name)
    
    # 如果是新的圖像名稱或到達列表末尾，處理並保存之前的图像
    if image_name != last_image_name and np_images_to_merge:
        # 合并图像
        final_np_img = np.sum(np_images_to_merge, axis=0)
        np.clip(final_np_img, 0, 255, out=final_np_img)  # 確保像素值在合理範圍内
        final_img = Image.fromarray(final_np_img.astype(np.uint8))
        
        # 保存图像
        final_img.save(base_save_path / f"{last_image_name}.png")
        print(f"圖像以保存到 {base_save_path / f'{last_image_name}.png'}")
        
        # 重置合并列表
        np_images_to_merge = []
    
    # 添加當前修改後的遮罩到合併列表
    np_images_to_merge.append(modified_np_mask)
    last_image_name = image_name

# 處理列表中的最後一個圖像
if np_images_to_merge:
    final_np_img = np.sum(np_images_to_merge, axis=0)
    np.clip(final_np_img, 0, 255, out=final_np_img)
    final_img = Image.fromarray(final_np_img.astype(np.uint8))
    final_img.save(base_save_path / f"{last_image_name}.png")
    print(f"圖像已保存到 {base_save_path / f'{last_image_name}.png'}")
  
###============================================================================###

###============================================================================###
# 因為是要灰度圖故從以上得到之結果，根據以下映射成灰度圖
# 定義顏色到灰度之映射

# color_to_gray = {

      (0, 0, 0): 0,         # 黑色 -> 背景

      (255, 255, 0): 1,    # 黄色 -> AWE

      (255, 0, 0): 2,      # 红色 -> Punctation

      (0, 255, 0): 3,      # 绿色 -> Mosaic

      (0, 0, 255): 4       # 蓝色 -> Atypical
    
# }
import os
from PIL import Image
import numpy as np

# 输入和输出文件夹的路径
input_folder_path = r'输入文件夹的路径'  
output_folder_path = r'輸出文件夹的路径'

# 确保输出文件夹存在
os.makedirs(output_folder_path, exist_ok=True)

# 定义颜色到灰度值的映射
color_to_gray = {
    (0, 0, 0): 0,         # 黑色 -> 背景
    (255, 255, 0): 1,    # 黄色 -> AWE
    (255, 0, 0): 2,      # 红色 -> Punctation
    (0, 255, 0): 3,      # 绿色 -> Mosaic
    (0, 0, 255): 4       # 蓝色 -> Atypical
}

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder_path):
    if filename.endswith(".png"):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder_path, filename)

        # 加载图像并转换为RGB
        image = Image.open(file_path).convert('RGB')
        image_array = np.array(image)

        # 初始化一个灰度图像数组
        gray_image_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

        # 遍历映射表，更新灰度图像数组
        for rgb, gray in color_to_gray.items():
            # 寻找当前颜色匹配的所有像素
            matches = (image_array == rgb).all(axis=-1)
            # 将匹配的像素在灰度图像中设置为相应的灰度值
            gray_image_array[matches] = gray

        # 创建灰度图像
        gray_image = Image.fromarray(gray_image_array)

        # 保存灰度图像到输出文件夹，文件名与原文件相同
        output_file_path = os.path.join(output_folder_path, filename)
        gray_image.save(output_file_path)

print("圖像轉換完成並保存。")
###============================================================================###

###============================================================================###
import shutil
from pathlib import Path

# 源文件夾路徑、目標文件夾路徑
source_folder = Path(r'源文件夾路徑') 
target_folder = Path(r'目標文件夾路徑')


# 确保目标文件夹存在
target_folder.mkdir(parents=True, exist_ok=True)

# 遍历 id_urls 数组
for image_id in id_urls:
    # 构建源文件和目标文件的完整路径
    source_file = source_folder / f"{image_id}.jpg"
    target_file = target_folder / f"{image_id}.jpg"
    
    # 复制文件
    try:
        shutil.copy(source_file, target_file)
        print(f"文件 {source_file} 已成功複製到 {target_file}")
    except FileNotFoundError:
        print(f"未找到文件：{source_file}")
    except Exception as e:
        print(f"複製文件發生錯誤：{e}")
###============================================================================###

###============================================================================###
# 增量 :
# 原圖、隨機左轉、隨機右轉、高斯模糊、調亮、調暗、X軸平移、Y軸平移。
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
import matplotlib.pyplot as plt
import os

def process_image(image_path):
    # 原图
    img = Image.open(image_path)
    img = img.resize((512, 512))  # 確保圖像大小为512x512

    plt.figure(figsize=(12, 8))
    
    """
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    """
    # 隨機左旋轉 0~45度
    angle = random.randint(1, 45)
    left_angle = angle
    rotated_img_left = img.rotate(angle, expand=False, fillcolor='black')
    """
    plt.subplot(2, 4, 2)
    plt.imshow(rotated_img_left)
    plt.title('Rotate Left')
    plt.axis('off')
    """
    # 隨機右旋轉 0~45度
    angle = random.randint(-45, -1)
    right_angle = angle
    rotated_img_right = img.rotate(angle, expand=False, fillcolor='black')
    """
    plt.subplot(2, 4, 3)
    plt.imshow(rotated_img_right)
    plt.title('Rotate Right')
    plt.axis('off')
    """
    # 高斯模糊
    blurred_img = img.filter(ImageFilter.GaussianBlur(5))
    """
    plt.subplot(2, 4, 4)
    plt.imshow(blurred_img)
    plt.title('Gaussian Blur')
    plt.axis('off')
    """
    # 調亮15%
    enhancer = ImageEnhance.Brightness(img)
    brightened_img = enhancer.enhance(1.15)
    """
    plt.subplot(2, 4, 5)
    plt.imshow(brightened_img)
    plt.title('Brightened 15%')
    plt.axis('off')
    """
    
    # 調暗15%
    darkened_img = enhancer.enhance(0.85)
    """
    plt.subplot(2, 4, 6)
    plt.imshow(darkened_img)
    plt.title('Darkened 15%')
    plt.axis('off')
    """
    
    # X軸 隨機平移10個pixel，保持圖像大小為512x512
    if random.random() > 0.5:
        x_translate = random.randint(-30, -20)
    else:
        x_translate = random.randint(20, 30)
    x_move = x_translate
    translated_img_x = Image.new("RGB", (512, 512), "black")
    translated_img_x.paste(img, (x_translate, 0))
    """
    plt.subplot(2, 4, 7)
    plt.imshow(translated_img_x)
    plt.title('Random X Translation')
    plt.axis('off')
    """
    
    # Y軸 隨機平移10個pixel，保持圖像大小為512x512
    if random.random() > 0.5:
        y_translate = random.randint(-30, -20)
    else:
        y_translate = random.randint(20, 30)
    y_move = y_translate
    translated_img_y = Image.new("RGB", (512, 512), "black")
    # 為向上平移和向下平移都正確設置貼上位置
    paste_position = (0, max(0, y_translate)) if y_translate > 0 else (0, y_translate)
    translated_img_y.paste(img, paste_position)
    """
    plt.subplot(2, 4, 8)
    plt.imshow(translated_img_y)
    plt.title('Random Y Translation')
    plt.axis('off')
    """
    plt.tight_layout()
    plt.show()
    return left_angle, right_angle, x_move, y_move, \
        [img, rotated_img_left, rotated_img_right, blurred_img, \
        brightened_img, darkened_img, translated_img_x, translated_img_y] 
###============================================================================###

###============================================================================###
# 紀錄當前編號
import os

def save_jpg_filenames_to_file(directory_path, output_file_path):
    """
    將指定目錄下的所有.jpg檔案名儲存到一個文本檔案中，每個檔案名佔一行。

    參數:
    - directory_path: 包含.jpg檔案的目錄路徑。
    - output_file_path: 儲存檔案名的文本檔案路徑。
    """
    # 獲取目錄下所有.jpg檔案的檔案名（不包括擴展名）
    filenames = [os.path.splitext(f)[0] for f in os.listdir(directory_path) if f.endswith('.jpg')]
    
    # 將檔案名寫入指定的文本檔案，每個檔案名佔一行
    with open(output_file_path, 'w') as file:
        for filename in filenames:
            file.write(filename + '\n')

# 範例使用
# 注意：請在本地環境中執行此代碼，並替換以下路徑為實際的目錄和輸出檔案路徑
directory_path = r'圖像路徑'
output_file_path = r'圖像所有的編號\.txt'
save_jpg_filenames_to_file(directory_path, output_file_path)
###============================================================================###

###============================================================================###
# 隨機分配1~700名編號，按8:1:1的比例，分配給train、val、test
import os
import random

# 設定編號範圍
ids = list(range(1, 701))

# 打亂編號順序
random.shuffle(ids)

# 按 8:1:1 的比例分配
num_train = int(0.8 * len(ids))
num_val = int(0.1 * len(ids))
num_test = len(ids) - num_train - num_val

train_ids = ids[:num_train]
val_ids = ids[num_train:num_train + num_val]
test_ids = ids[num_train + num_val:]

# 檔案路徑
train_path = r'\train.txt'
val_path = r'\val.txt'
test_path = r'\test.txt'

# 確保目標目錄存在
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(val_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)

# 寫入文件
with open(train_path, 'w') as f:
    for id in train_ids:
        f.write(f'{id}\n')

with open(val_path, 'w') as f:
    for id in val_ids:
        f.write(f'{id}\n')

with open(test_path, 'w') as f:
    for id in test_ids:
        f.write(f'{id}\n')

print("已完成編號分配並寫入文件。")
###============================================================================###

###============================================================================###
# 從 train、val.txt 中提取編號影像並將其進行8倍增量，並儲存(原彩色影像)。
# 00001.jpg ~ 00008.jpg
# 00009.jpg ~ 00016.jpg
# .
# 00673.jpg ~ 00680.jpg
def read_image_ids(txt_file_path):
    """從.txt文件中讀取圖像編號"""
    with open(txt_file_path, 'r') as file:
        image_ids = file.read().splitlines()
    return image_ids

# 設置.txt文件和圖像目錄的路径
txt_file_path = r'train、val.txt'
images_directory_path = r'原影像路徑\JPEGImages'
output_directory_path = r'增量後影像路徑\JPEGImages(aug)'  # 輸出資料夾

# 確保输出目錄存在
os.makedirs(output_directory_path, exist_ok=True)
# 圖項文件編號
img_num = 1

# 讀取圖像目錄
image_ids = read_image_ids(txt_file_path)

all_left_angle=[]
all_right_angle=[]
all_x_move=[]
all_y_move=[]

# 對於每個圖像編號，找到對應的 .jpg 文件並處理
for image_id in image_ids:
    image_path = os.path.join(images_directory_path, f'{image_id}.jpg')
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        #process_image(image_path)
        left_angle, right_angle, x_move, y_move, processed_images = process_image(image_path)
        
        # 紀錄左右旋轉角度、上下平移
        all_left_angle.append(left_angle)
        all_right_angle.append(right_angle)
        all_x_move.append(x_move)
        all_y_move.append(y_move)

        # 保存處理後的圖像
        for img in processed_images:
            # 構建輸出文件名，格式為：00001.jpg, 00002.jpg, ...
            output_filename = f'{img_num:05d}.jpg'
            output_path = os.path.join(output_directory_path, output_filename)
            img.save(output_path)
            #print(f"Saved: {output_path}")
            img_num += 1  # 更新圖像編號

    else:
        print(f"Image file not found: {image_path}")
###============================================================================###

###============================================================================###
# 從 Train、val.txt 中提取編號影像並將其進行8倍增量，並儲存(原遮罩影像)。
# 00001.png ~ 00008.png
# 00009.png ~ 00016.png
# .
# 00673.png ~ 00680.png
from PIL import Image, ImageOps
import os

# 
# all_left_angle 、 all_right_angle 、 all_x_move 、 all_y_move

# 函数：根據給定參數處理影像
def process_and_save_images(image_id, output_dir, img_num):
    image_path = os.path.join(images_directory_path, f'{image_id}.png')
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return img_num
    
    img = Image.open(image_path)
    
    # 處理影像
    images = [img]  # 原圖
    images.append(img.rotate(-all_left_angle[img_num // 8]))  # 左轉   //(整除)
    images.append(img.rotate(all_right_angle[img_num // 8]))  # 右轉
    images += [img] * 3  # 再添加三次原圖
    # X軸平移
    translated_img_x = ImageOps.expand(img, border=(all_x_move[img_num // 8], 0), fill='black')
    images.append(ImageOps.crop(translated_img_x, border=(-all_x_move[img_num // 8], 0)))
    # Y軸平移
    translated_img_y = ImageOps.expand(img, border=(0, all_y_move[img_num // 8]), fill='black')
    images.append(ImageOps.crop(translated_img_y, border=(0, -all_y_move[img_num // 8])))
    
    # 保存處理後的影像
    for processed_img in images:
        output_filename = f'{img_num:05d}.png'
        processed_img.save(os.path.join(output_dir, output_filename))
        img_num += 1
    
    return img_num

# 讀取圖像編號
txt_file_path = r'\train、val.txt'
images_directory_path = r'\SegmentationClass'
output_directory = r'\SegmentationClass(aug)'
os.makedirs(output_directory, exist_ok=True)

with open(txt_file_path, 'r') as file:
    image_ids = file.read().splitlines()

img_num = 1
for image_id in image_ids:
    img_num = process_and_save_images(image_id, output_directory, img_num)
###============================================================================###



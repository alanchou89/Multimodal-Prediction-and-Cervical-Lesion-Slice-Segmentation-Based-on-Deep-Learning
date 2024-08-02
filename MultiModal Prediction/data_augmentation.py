# 再.ipynb上分段運行

###================================================================================###
# 亮度均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def equalize_histogram_image(image_path, return_image=False):
    """
    對.tif格式的彩色陰道鏡影像進行直方圖均衡化。

    :param image_path: 影像的路徑
    :param return_image: 是否返回處理後的影像。如果為False，則顯示原始影像和處理後的影像。
    :return: 根據return_image參數，可能返回處理後的影像，或無返回值但顯示影像。
    """
    # 讀取彩色影像
    img = cv2.imread(image_path)

    # 檢查影像是否成功加載
    if img is None:
        print("無法加載影像:", image_path)
        return None if return_image else None

    # 將BGR影像轉換到YCrCb色彩空間
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 對Y通道（亮度）進行直方圖均衡化
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    # 將處理後的影像轉換回BGR色彩空間
    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    # 獲取並打印影像名稱
    image_name = os.path.basename(image_path)
    print("正在處理影像:", image_name)

    if return_image:
        # 返回處理後的影像
        return cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB)
    else:
        # 顯示原始影像和處理後的影像
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB))
        plt.title("Equalized Image")
        plt.axis('off')

        plt.show()
###================================================================================###

###================================================================================###
# 水平、垂直翻轉圖像
import cv2
import matplotlib.pyplot as plt

def flip_image_horizontally_vertically(image_path):
    """
    顯示原始圖像以及水平、垂直翻轉圖像
    """
    # 讀取彩色影像
    img = cv2.imread(image_path)

    # 檢查是否成功讀取
    if img is None:
        print("無法讀取圖像:", image_path)
        return

    # 水平翻轉圖像
    img_flipped_horizontally = cv2.flip(img, 1)

    # 垂直翻轉圖像
    img_flipped_vertically = cv2.flip(img, 0)

    # 印出原始圖像、水平翻轉圖像、垂直翻轉圖像
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_flipped_horizontally, cv2.COLOR_BGR2RGB))
    plt.title("Horizontally Flipped Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_flipped_vertically, cv2.COLOR_BGR2RGB))
    plt.title("Vertically Flipped Image")
    plt.axis('off')

    plt.show()
###================================================================================###

###================================================================================###
# 隨機旋轉
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image_path, angle):
    # 讀取彩色影像
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # 計算旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 計算新邊界長寬
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 調整旋轉矩陣並考慮平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # 旋轉
    rotated = cv2.warpAffine(img, M, (nW, nH))

    # 結果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title("Rotated Image")
    plt.axis('off')

    plt.show()
###================================================================================###

###================================================================================###
# 將excel中的編號讀取，並按造3:1:1的比例分配給train、val、test
import pandas as pd
import numpy as np
import random
import os

# 讀取 Excel 文件
file_path = r'Excel 文件位置' 
df = pd.read_excel(file_path)

# 提取編號
ids = df['編號'].tolist()

# 隨機打亂編號
random.shuffle(ids)

# 分配比例 3:1:1
total = len(ids)
train_size = int(total * 0.6)
val_size = int(total * 0.2)
test_size = total - train_size - val_size

train_ids = ids[:train_size]
val_ids = ids[train_size:train_size + val_size]
test_ids = ids[train_size + val_size:]

# 定義目標路徑
train_path = r'C:\GitHub\dataset\train\train.txt'
val_path = r'C:\GitHub\dataset\val\val.txt'
test_path = r'C:\GitHub\dataset\test\test.txt'

# 確保目標目錄存在
os.makedirs(os.path.dirname(train_path), exist_ok=True)
os.makedirs(os.path.dirname(val_path), exist_ok=True)
os.makedirs(os.path.dirname(test_path), exist_ok=True)

# 將結果寫入文件
with open(train_path, 'w') as f:
    for id in train_ids:
        f.write(f"{id}\n")

with open(val_path, 'w') as f:
    for id in val_ids:
        f.write(f"{id}\n")

with open(test_path, 'w') as f:
    for id in test_ids:
        f.write(f"{id}\n")

print("分配完成")
###================================================================================###

###================================================================================###
# 對 train 、 val 增量的方法
import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_and_augment_image(image_path,angle = None):
    """
    對.tif 格式的彩色陰道鏡影像進行直方圖均衡化，並執行水平翻轉、垂直翻轉、旋轉和高斯濾波處理。
    """
    # 讀取彩色影像
    img = cv2.imread(image_path)

    # 檢查是否成功讀取
    if img is None:
        print("無法讀取圖像:", image_path)
        return

    # 將影像缩放到512x512大小
    img = cv2.resize(img, (200, 200))

    # 將 BGR 影像轉換到 YCrCb 色彩空間
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 對 Y 通道（亮度）進行直方圖均衡化
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])

    # 將處理後的影像轉換回 BGR 色彩空間
    img_equalized = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

    # 水平翻轉
    flipped_horizontally = cv2.flip(img_equalized, 1)

    # 垂直翻轉
    flipped_vertically = cv2.flip(img_equalized, 0)

    # 旋轉
    (h, w) = img_equalized.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转45度
    rotated = cv2.warpAffine(img_equalized, M, (w, h))

    # 高斯滤波
    gaussian_blurred = cv2.GaussianBlur(img_equalized, (5, 5), 0)

    # 印出原始影像和增量影像
    plt.figure(figsize=(15, 6))
   
    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(flipped_horizontally, cv2.COLOR_BGR2RGB))
    plt.title("Horizontally")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(flipped_vertically, cv2.COLOR_BGR2RGB))
    plt.title("Vertically")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB))
    plt.title("GaussianBlur")
    plt.axis('off')


    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title("Rotated")
    plt.axis('off')
    

    plt.show()

equalize_and_augment_image(r"N 影像位置\.tif", angle = 45)
equalize_and_augment_image(r"A 影像位置\.tif", angle = 45)
equalize_and_augment_image(r"L 影像位置\.tif", angle = 45)
###================================================================================###

### 再.ipynb上分段運行

###=======================================================================================###
import json
import torch
import joblib
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from torchvision import models, transforms
from PIL import Image
import os
import joblib
import torch.nn as nn
###=======================================================================================###

###=======================================================================================###
# 檢查是否有可用的 GPU，如果有，使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
###=======================================================================================###

###=======================================================================================###
# patient_ids存放所有病人編號
# 指定Excel文件的路徑
excel_file_path = r'指定Excel文件的路徑'

# 使用Pandas讀取Excel文件，假設第一行是列名
df = pd.read_excel(excel_file_path)

# 提取所有病人編號，假設編號在'A'列
patient_ids = df['編號'].tolist()
patient_ids
###=======================================================================================###

###=======================================================================================###
# patient_diagnosis_type 存放病人編號和診斷結果
# 讀取數據
df = pd.read_excel(r'指定Excel文件的路徑') 

# 建立診斷類別到數字的映射
diagnosis_mapping = {
    'CIS': 0,
    'Mild dysplasia': 1,
    'Moderate dysplasia': 2,
    'Severe dysplasia': 3
}

# 將診斷類別轉換為數字
df['diagnosis_numeric'] = df['mostly diagnosis'].map(diagnosis_mapping)

# 創建病人診斷類別字典
patient_diagnosis_type = df.set_index('編號')['diagnosis_numeric'].to_dict()
###=======================================================================================###

###=======================================================================================###
# 處理年齡資料
# 這個文件路徑應該替換為您Excel文件的實際路徑
excel_file_path = r'指定Excel文件的路徑'

# 使用Pandas讀取Excel文件，假設列A是病人編號，列B是年齡，第一行是列名
df = pd.read_excel(excel_file_path, header=0)
# 
df.columns = ['編號', '年齡','mostly diagnosis']

# 將病人編號設為索引
df.set_index('編號', inplace=True)

# 標準化年齡數據
max_age = df['年齡'].max()
min_age = df['年齡'].min()
df['Normalized_Age'] = (df['年齡'] - (max_age + min_age) / 2) / (max_age - min_age) * 2

# 將標準化的年齡數據轉換為字典
t_cli_feas = df['Normalized_Age'].to_dict()

# 定義一個函數將年齡轉換為獨熱編碼
def age_to_one_hot(age, num_categories=20, vector_length=1024):
    category = int(age // 5)  # 將年齡分為5年一個區間
    one_hot = np.zeros(num_categories)
    one_hot[category] = 1
    # 重複向量直到接近目標長度，然後截取到精確長度
    repeated_vector = np.tile(one_hot, (vector_length // num_categories) + 1)
    return repeated_vector[:vector_length]

# 沒有標準化
# 應用這個函數到DataFrame
df['One_Hot_Age'] = df['年齡'].apply(age_to_one_hot)
# 將結果轉換為字典
one_hot_age_dict = df['One_Hot_Age'].to_dict()

# 標準化
# 應用這個函數到DataFrame
df['One_Hot_Age_std'] = df['Normalized_Age'].apply(age_to_one_hot)
# 將結果轉換為字典
one_hot_age_dict_std = df['One_Hot_Age_std'].to_dict()


# 沒有標準化
# 定義嵌入層，其中num_embeddings設置為年齡的最大值加1（確保年齡最大值也被包括）
max_age = df['年齡'].max()
age_embedding = nn.Embedding(num_embeddings=max_age+1, embedding_dim=1024)
# 將年齡轉換為嵌入向量
df['Age_Embedding'] = df['年齡'].apply(lambda x: age_embedding(torch.tensor([x])).detach().numpy())
# 轉換為字典
age_embeddings = df['Age_Embedding'].to_dict()

# 標準化
# 定義嵌入層
age_embedding = nn.Embedding(num_embeddings=101, embedding_dim=1024)
# 將標準化年齡轉換為嵌入向量
df['Age_Embedding_std'] = df['Normalized_Age'].apply(lambda x: age_embedding(torch.tensor([int((x + 1) / 2 * 100)])).detach().numpy())
# 轉換為字典
age_embeddings_std = df['Age_Embedding_std'].to_dict()

# 結合 one_hot_age_dict 和 age_embeddings
cli_embeddings = {}

for patient_id in df.index:
    one_hot_vector = one_hot_age_dict[patient_id]
    one_hot_vector_std = one_hot_age_dict_std[patient_id]
    embedding_vector = age_embeddings[patient_id]
    embedding_vector_std = age_embeddings_std[patient_id]
    
    combined_vector = np.vstack([one_hot_vector,one_hot_vector_std, embedding_vector,embedding_vector_std])
    cli_embeddings[patient_id] = combined_vector
###=======================================================================================###

###=======================================================================================###
# 處理圖像資料
# 加載預訓練的 ResNet50 或 ResNet101 模型
"""
model = models.resnet50(pretrained=True)  # 50
"""
model = models.resnet101(pretrained=True)  

# 修改最後一個全連接層，使其輸出長度為 1024 的特徵向量
model.fc = nn.Linear(model.fc.in_features, 1024)
# 將模型移至 GPU
model = model.to(device)
model.eval()

# 設置圖像轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 圖像放大和分割函數
def resize_and_split_image(image_path, target_size=1024, split_size=256):
    with Image.open(image_path) as img:
        # 放大圖像到 target_size
        img_resized = img.resize((target_size, target_size), Image.BILINEAR)
        # 分割圖像為 split_size 的小塊
        patches = []
        for i in range(0, target_size, split_size):
            for j in range(0, target_size, split_size):
                patch = img_resized.crop((i, j, i + split_size, j + split_size))
                patches.append(patch)
        return patches

# 特徵提取函數
def extract_features(image, model, transform, device):
    image = transform(image).to(device)
    with torch.no_grad():
        features = model(image.unsqueeze(0))
    return features.cpu().numpy().flatten()

# 特徵字典(N)
features_dict = {}
# 已處理的病人編號集合
processed_patients = set()
# 遍歷影像資料夾
image_folder = r'影像資料夾'
num_patients_processed = 0  # 添加一個計數器來跟踪處理的病人數量

for image_file in os.listdir(image_folder):
    # 檢查影像文件是否屬於指定的病人編號且為一般影像（'N'）
    if 'N' in image_file:
        # 提取病人編號，移除前導零
        patient_id = int(image_file.split('N')[0])
        # 確保每個病人編號只處理一次
        if patient_id in patient_ids and patient_id not in processed_patients:
            processed_patients.add(patient_id)
            
            num_patients_processed += 1  # 增加計數器
            print(f"正在處理第 {num_patients_processed} 位病人，病人編號：{patient_id}")  # 打印訊息

            image_path = os.path.join(image_folder, image_file)
            # 放大並分割圖像
            patches = resize_and_split_image(image_path)
             # 提取每個小塊的特徵
            patch_features_dict = {}

            for idx, patch in enumerate(patches):
                # 為每個切片創建一個唯一的文件名
                patch_file_name = f"{patient_id}-{idx}_N.png"
                features = extract_features(patch, model, transform, device)
                patch_features_dict[patch_file_name] = features
            features_dict[patient_id] = patch_features_dict

# 保存特徵字典為 .pkl 文件
output_path = r'N.pkl 文件存放位置'
joblib.dump(features_dict, output_path)

# 特徵字典(A)
features_dict = {}
# 已處理的病人編號集合
processed_patients = set()
# 遍歷影像資料夾
image_folder = r'影像資料夾'
num_patients_processed = 0  # 添加一個計數器來跟踪處理的病人數量

for image_file in os.listdir(image_folder):
    # 檢查影像文件是否屬於指定的病人編號且為一般影像（'A'）
    if 'A' in image_file:
        # 提取病人編號，移除前導零
        patient_id = int(image_file.split('A')[0])
        # 確保每個病人編號只處理一次
        if patient_id in patient_ids and patient_id not in processed_patients:
            processed_patients.add(patient_id)
            
            num_patients_processed += 1  # 增加計數器
            print(f"正在處理第 {num_patients_processed} 位病人，病人編號：{patient_id}")  # 打印信息

            image_path = os.path.join(image_folder, image_file)
            # 放大並分割圖像
            patches = resize_and_split_image(image_path)
            # 提取每個小塊的特徵
            patch_features_dict = {}

            for idx, patch in enumerate(patches):
                # 為每個切片創建一個唯一的文件名
                patch_file_name = f"{patient_id}-{idx}_A.png"
                features = extract_features(patch, model, transform, device)
                patch_features_dict[patch_file_name] = features
            features_dict[patient_id] = patch_features_dict

# 保存特徵字典為 .pkl 文件
output_path = r'A.pkl 文件存放位置'
joblib.dump(features_dict, output_path)

# 特徵字典(L)
features_dict = {}
# 已處理的病人編號集合
processed_patients = set()
# 遍歷影像資料夾
image_folder = r'影像資料夾'
num_patients_processed = 0  # 添加一個計數器來跟踪處理的病人數量

for image_file in os.listdir(image_folder):
    # 檢查影像文件是否屬於指定的病人編號且為一般影像（'L'）
    if 'L' in image_file:
        # 提取病人編號，移除前導零
        patient_id = int(image_file.split('L')[0])
        # 確保每個病人編號只處理一次
        if patient_id in patient_ids and patient_id not in processed_patients:
            processed_patients.add(patient_id)
            
            num_patients_processed += 1  # 增加計數器
            print(f"正在處理第 {num_patients_processed} 位病人，病人編號：{patient_id}")  # 打印信息

            image_path = os.path.join(image_folder, image_file)
            # 放大並分割圖像
            patches = resize_and_split_image(image_path)
            # 提取每個小塊的特徵\
            patch_features_dict = {}

            for idx, patch in enumerate(patches):
                # 為每個切片創建一個唯一的文件名
                patch_file_name = f"{patient_id}-{idx}_L.png"
                features = extract_features(patch, model, transform, device)
                patch_features_dict[patch_file_name] = features
            features_dict[patient_id] = patch_features_dict

# 保存特徵字典為 .pkl 文件
output_path = r'L.pkl 文件存放位置'
joblib.dump(features_dict, output_path)

t_img_N_fea = joblib.load(r'N.pkl 文件存放位置')
t_img_A_fea = joblib.load(r'A.pkl 文件存放位置')
t_img_L_fea = joblib.load(r'L.pkl 文件存放位置')
###=======================================================================================###

###=======================================================================================###
# Construction of Multimodal Graphs (多模態圖的構建)
feature_img_N = {}
feature_img_A = {}
feature_img_L = {}
feature_cli = {}
data_type = {}
for x in patient_ids:
    f_img_N = []
    f_img_A = []
    f_img_L = []
    f_cli = []
    t_type = []
    if x in t_img_N_fea:
        for z in t_img_N_fea[x]:
            f_img_N.append(t_img_N_fea[x][z])
        t_type.append('imgN')
    
    if x in t_img_A_fea:
        for z in t_img_A_fea[x]:
            f_img_A.append(t_img_A_fea[x][z])
        t_type.append('imgA')
    
    if x in t_img_L_fea:
        for z in t_img_L_fea[x]:
            f_img_L.append(t_img_L_fea[x][z])
        t_type.append('imgL')
    if x in  cli_embeddings:       
        for r in cli_embeddings[x]:
            f_cli.append(r) 
        t_type.append('cli')


    data_type[x]=t_type
    feature_img_N[x] = f_img_N
    feature_img_A[x] = f_img_A
    feature_img_L[x] = f_img_L
    feature_cli[x] = f_cli

def get_edge_index_image(id, image_type):
    start = []
    end = [] 
    
    if image_type == 'N':
        t_img_fea = t_img_N_fea
    elif image_type == 'A':
        t_img_fea = t_img_A_fea
    elif image_type == 'L':
        t_img_fea = t_img_L_fea

    if id in t_img_fea:
        #定義固定位置關係
        adjacency_relations = {
            0: [1, 4, 5],  
            1: [0, 2, 4, 5, 6],  
            2: [1, 3, 5, 6, 7],  
            3: [2, 6, 7],
            4: [0, 1, 5, 8, 9],  
            5: [0, 1, 2, 4, 6, 8, 9, 10],  
            6: [1, 2, 3, 5, 7, 9, 10, 11],  
            7: [2, 3, 6, 10, 11],   
            8: [4, 5, 9, 12, 13],  
            9: [4, 5, 6, 8, 10, 12, 13, 14],  
            10: [5, 6, 7, 9, 11, 13, 14 ,15],  
            11: [6, 7, 10, 14 ,15],    
            12: [8, 9, 13],  
            13: [8, 9, 10, 12, 14],  
            14: [9, 10, 11, 13, 15],  
            15: [10, 11, 14]
        }

        for patch_name in t_img_fea[id]:
            # 提取位置編號
            position = int(patch_name.split('-')[1].split('_')[0])
            # 根據預定義的連接關係添加邊
            for neighbor in adjacency_relations[position]:
                start.append(position)
                end.append(neighbor)

    return [start, end]
# 新的cli邊表示
def get_edge_index_cli(id):   
    start = []
    end = []   
    if id in cli_embeddings:
        for i in range(len(feature_cli[id])):
            for j in range(len(feature_cli[id])):
                if i!=j:
                    start.append(j)
                    end.append(i)
    return [start,end]  


all_data = {}
for id in patient_ids:
    print(id)
    node_img_N=torch.tensor(feature_img_N[id],dtype=torch.float) 
    node_img_A=torch.tensor(feature_img_A[id],dtype=torch.float) 
    node_img_L=torch.tensor(feature_img_L[id],dtype=torch.float) 
    node_cli=torch.tensor(feature_cli[id],dtype=torch.float)
    edge_index_image_N = torch.tensor(get_edge_index_image(id,image_type='N'),dtype=torch.long)
    edge_index_image_A = torch.tensor(get_edge_index_image(id,image_type='A'),dtype=torch.long)
    edge_index_image_L = torch.tensor(get_edge_index_image(id,image_type='L'),dtype=torch.long)
    edge_index_cli = torch.tensor(get_edge_index_cli(id),dtype=torch.long)
    diagnosis_type=torch.tensor([patient_diagnosis_type[id]])
    data_id = id 
    t_data_type = data_type[id]

    data=Data(x_imgN=node_img_N,x_imgA=node_img_A,x_imgL=node_img_L,x_cli=node_cli,diagnosis_type=diagnosis_type,data_id=data_id,data_type=t_data_type,
              edge_index_imageN=edge_index_image_N,edge_index_imageA=edge_index_image_A,edge_index_imageL=edge_index_image_L,edge_index_cli=edge_index_cli)
    all_data[id] = data
    print(data)

joblib.dump(all_data,r'存放圖結構資料位置')  # C:\ChiMei\MY_HGCN\all_data_101.pkl
all_data = joblib.load(r'存放圖結構資料位置')

all_data
###=======================================================================================###

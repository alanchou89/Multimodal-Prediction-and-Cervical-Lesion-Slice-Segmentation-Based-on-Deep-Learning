###===在ipynb 上分段運行===###
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
###===================###

###===================###
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import torch
import joblib
import random
import json
import math
import sys
import argparse
import numpy as np
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold 
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
#from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from my_mae_model import fusion_model_mae_2
from util import Logger,adjust_learning_rate
from mae_utils import generate_mask
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print("Python 解釋器路徑:", sys.executable)
print(os.environ['PATH'])
###===================###

###===================###
# 用於模型預測和評估的函數，主要用於處理驗證或測試數據集。
def prediction(all_data, v_model, val_id, patient_diagnosis_type, args):
    
    v_model.eval()
    v_model = v_model.to(device) ###
    val_diagnosis_T = []
    logits_all, logits_imgN, logits_imgA, logits_imgL, logits_cli = [], [], [], [], []
    # 定義交叉熵損失函數
    criterion = nn.CrossEntropyLoss()

    # 儲存預測的標籤
    predicted_labels_all = []
    predicted_labels_imgN = []
    predicted_labels_imgA = []
    predicted_labels_imgL = []
    predicted_labels_cli = []
    loss_all = 0.0

    ## 新增: 計算混淆矩陣
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for i_batch, id in enumerate(val_id):
            # 獲取圖
            graph = all_data[id].to(device)

            if args.train_use_type != None:
                use_type_eopch = args.train_use_type
            else:
                # 使用圖數據本身的 data_type 作為訓練數據的類型。
                use_type_eopch = graph.data_type

            out_pre,out_fea,out_att,_,\
            class_logits_all,class_logits_imgN,class_logits_imgA,class_logits_imgL,class_logits_cli \
            = v_model(graph,args.train_use_type,use_type_eopch,mix=args.mix)

            # [batch_size, num_classes]
            class_logits_all = class_logits_all.unsqueeze(0)  # 將尺寸從 [4] 轉換為 [1, 4]
            class_logits_imgN = class_logits_imgN.unsqueeze(0)
            class_logits_imgA = class_logits_imgA.unsqueeze(0)
            class_logits_imgL = class_logits_imgL.unsqueeze(0)
            class_logits_cli = class_logits_cli.unsqueeze(0)

            # 將 logits 轉換為概率
            probabilities_all = F.softmax(class_logits_all, dim=1) 
            probabilities_imgN = F.softmax(class_logits_imgN, dim=1) 
            probabilities_imgA = F.softmax(class_logits_imgA, dim=1)  
            probabilities_imgL = F.softmax(class_logits_imgL, dim=1)  
            probabilities_cli = F.softmax(class_logits_cli, dim=1)   
            # 獲得此病患預測標籤
            pred_label_all = torch.argmax(probabilities_all, dim=1).cpu().numpy()
            pred_label_imgN = torch.argmax(probabilities_imgN, dim=1).cpu().numpy()
            pred_label_imgA = torch.argmax(probabilities_imgA, dim=1).cpu().numpy()
            pred_label_imgL = torch.argmax(probabilities_imgL, dim=1).cpu().numpy()
            pred_label_cli = torch.argmax(probabilities_cli, dim=1).cpu().numpy()
            # 紀錄預測所有患者的預測標籤
            predicted_labels_all.append(pred_label_all)
            predicted_labels_imgN.append(pred_label_imgN)
            predicted_labels_imgA.append(pred_label_imgA)
            predicted_labels_imgL.append(pred_label_imgL)
            predicted_labels_cli.append(pred_label_cli)

            # 將此病患的病灶加入列表
            val_diagnosis_T.append(patient_diagnosis_type[id])
            # 添加logits到对应的list
            logits_all.append(class_logits_all)
            logits_imgN.append(class_logits_imgN)
            logits_imgA.append(class_logits_imgA)
            logits_imgL.append(class_logits_imgL)
            logits_cli.append(class_logits_cli)

            ## 新增: 計算混淆矩陣
            ## 紀錄預測標籤和真實標籤
            predicted_labels.append(pred_label_all)
            true_labels.append(patient_diagnosis_type[id])

   
    logits_all = torch.cat(logits_all, dim = 0)    # [val_id, 4]
    logits_imgN = torch.cat(logits_imgN, dim = 0)
    logits_imgA = torch.cat(logits_imgA, dim = 0)
    logits_imgL = torch.cat(logits_imgL, dim = 0)
    logits_cli = torch.cat(logits_cli, dim = 0)
    val_diagnosis_T = torch.tensor(val_diagnosis_T, dtype=torch.long).to(device)

    # 計算損失
    loss_all = criterion(logits_all, val_diagnosis_T)   # 此結果為平均損失
    loss = loss_all 

    # 轉換列表為 NumPy 數組
    predicted_labels_all = np.concatenate(predicted_labels_all)
    predicted_labels_imgN = np.concatenate(predicted_labels_imgN)
    predicted_labels_imgA = np.concatenate(predicted_labels_imgA)
    predicted_labels_imgL = np.concatenate(predicted_labels_imgL)
    predicted_labels_cli = np.concatenate(predicted_labels_cli)

    val_diagnosis_T_cpu = val_diagnosis_T.cpu().numpy()

    # 計算效能指標
    accuracy_all = accuracy_score(val_diagnosis_T_cpu, predicted_labels_all)
    accuracy_imgN = accuracy_score(val_diagnosis_T_cpu, predicted_labels_imgN)
    accuracy_imgA = accuracy_score(val_diagnosis_T_cpu, predicted_labels_imgA)
    accuracy_imgL = accuracy_score(val_diagnosis_T_cpu, predicted_labels_imgL)
    accuracy_cli = accuracy_score(val_diagnosis_T_cpu, predicted_labels_cli)

    ## 新增: 計算混淆矩陣
    ## 轉換真實標籤為Numpy數組
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
  
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    # 計算每個類別之評價指標
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    # 根據混淆矩陣計算偽陰性（FP）、偽陽性（FN）、敏感性和特異性
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # sensitivity = TP / (TP + FN)
    # specificity = TN / (TN + FP)

    return loss, accuracy_all, accuracy_imgN, accuracy_imgA, accuracy_imgL, accuracy_cli, \
        cm, precision, recall, f1, FP, FN, TP, TN, # sensitivity, specificity
###===================###

###===================###
def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
###===================###

###===================###
def train_a_epoch(model,train_data,all_data,patient_diagnosis_type,batch_size,optimizer,epoch,format_of_coxloss,args):
    model.train() 

    model = model.to(device)

    iter = 0
    loss_nn_all = [] 
 
    mes_loss_of_mae = nn.MSELoss()

    mse_loss_of_mae = 0.0
    loss_surv = 0.0

    # 定義交叉熵損失函數
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_samples = 0

    # 儲存預測的標籤
    predicted_labels_all = []
    predicted_labels_imgN = []
    predicted_labels_imgA = []
    predicted_labels_imgL = []
    predicted_labels_cli = []

    diagnosis_all = []
    loss_all = 0.0
    loss_imgN = 0.0
    loss_imgA = 0.0
    loss_imgL = 0.0
    loss_cli = 0.0

    logits_all, logits_imgN, logits_imgA, logits_imgL, logits_cli = [], [], [], [], []
    diagnosis_batch_T = []

    ## 新增: 計算混淆矩陣
    predicted_labels = []
    true_labels = []

    # 遍歷訓練數據集
    for i_batch,id in enumerate(train_data):
        iter += 1 
        # 獲取此病患使用多少模態
        num_of_modal = len(all_data[id].data_type)
        # 生成遮罩, 用於在訓練過程中隱藏部分數據
        mask = generate_mask(num=len(args.train_use_type))

        if args.train_use_type!=None:
            # use_type_eopch得到模態類型
            use_type_eopch = args.train_use_type
            num_of_modal = len(use_type_eopch)                
        else:
            use_type_eopch = all_data[id].data_type    
        graph = all_data[id].to(device)
        
        # 將此病患的病灶加入列表
        diagnosis_all.append(patient_diagnosis_type[id])

        out_pre,out_fea,out_att,fea_dict,\
            class_logits_all,class_logits_imgN,class_logits_imgA,class_logits_imgL,class_logits_cli \
            = model(graph,use_type_eopch,use_type_eopch,mask,mix=args.mix)
  
        # 參數被設為T            
        if args.add_mse_loss_of_mae:
            # 計算 MAE 輸出和目標標籤之間的 MSE 損失，並根據指定的因子將其加到總 MSE 損失上。
            mse_loss_of_mae += args.mse_loss_of_mae_factor * mes_loss_of_mae(input=fea_dict['mae_out'][mask[0]], target=fea_dict['mae_labels'][mask[0]])
   

        # [batch_size, num_classes]
        class_logits_all = class_logits_all.unsqueeze(0)  #加的
        class_logits_imgN = class_logits_imgN.unsqueeze(0) #加的
        class_logits_imgA = class_logits_imgA.unsqueeze(0) #加的
        class_logits_imgL = class_logits_imgL.unsqueeze(0) #加的
        class_logits_cli = class_logits_cli.unsqueeze(0)   #加的

        # 將 logits 轉換為概率
        probabilities_all = F.softmax(class_logits_all, dim=1) 
        probabilities_imgN = F.softmax(class_logits_imgN, dim=1) 
        probabilities_imgA = F.softmax(class_logits_imgA, dim=1)  
        probabilities_imgL = F.softmax(class_logits_imgL, dim=1)  
        probabilities_cli = F.softmax(class_logits_cli, dim=1)   
        # 獲得預測標籤
        pred_label_all = torch.argmax(probabilities_all, dim=1).cpu().numpy()
        pred_label_imgN = torch.argmax(probabilities_imgN, dim=1).cpu().numpy()
        pred_label_imgA = torch.argmax(probabilities_imgA, dim=1).cpu().numpy()
        pred_label_imgL = torch.argmax(probabilities_imgL, dim=1).cpu().numpy()
        pred_label_cli = torch.argmax(probabilities_cli, dim=1).cpu().numpy()
            
        predicted_labels_all.append(pred_label_all)
        predicted_labels_imgN.append(pred_label_imgN)
        predicted_labels_imgA.append(pred_label_imgA)
        predicted_labels_imgL.append(pred_label_imgL)
        predicted_labels_cli.append(pred_label_cli)

        # 將此病患的病灶加入列表
        diagnosis_batch_T.append(patient_diagnosis_type[id])
        # 添加logits到对应的list
        logits_all.append(class_logits_all)
        logits_imgN.append(class_logits_imgN)
        logits_imgA.append(class_logits_imgA)
        logits_imgL.append(class_logits_imgL)
        logits_cli.append(class_logits_cli)

        ## 新增: 計算混淆矩陣
        ## 紀錄預測標籤和真實標籤
        predicted_labels.extend(pred_label_all)
        true_labels.append(patient_diagnosis_type[id])

        # 檢查是否達到批次處理的末尾 或 到達數據集末尾
        if iter % batch_size == 0 or i_batch == len(train_data)-1:

            optimizer.zero_grad() 
            loss_surv = 0.0 
            loss_surv_imgN = 0.0
            loss_surv_imgA = 0.0
            loss_surv_imgL = 0.0
            loss_surv_cli = 0.0
            loss_surv_all = 0.0
       
            a = 0.0
            
            logits_all = torch.cat(logits_all, dim=0)
            logits_imgN = torch.cat(logits_imgN, dim=0)
            logits_imgA = torch.cat(logits_imgA, dim=0)
            logits_imgL = torch.cat(logits_imgL, dim=0)
            logits_cli = torch.cat(logits_cli, dim=0)
            diagnosis_batch_T = torch.tensor(diagnosis_batch_T, dtype=torch.long).to(device)
            #print("logits_imgN: ", logits_imgN) #[batch_size, 4]
        
            # 計算損失
            loss_all = criterion(logits_all, diagnosis_batch_T)   # 此結果為平均損失
            loss_imgN = criterion(logits_imgN, diagnosis_batch_T)
            loss_imgA = criterion(logits_imgA, diagnosis_batch_T)
            loss_imgL = criterion(logits_imgL, diagnosis_batch_T)
            loss_cli = criterion(logits_cli, diagnosis_batch_T)
            

            if format_of_coxloss == 'multi':
                loss_surv_imgN = ( loss_imgN ) * 0.3
                loss_surv_imgA = ( loss_imgA ) * 0.3
                loss_surv_imgL = ( loss_imgL ) * 0.3
                loss_surv_cli = ( loss_cli ) * 0.2   #0.2
                loss_surv_all = loss_all * 1.0 # 1.0
                loss_surv = loss_surv_imgN + loss_surv_imgA + loss_surv_imgL + loss_surv_cli + loss_surv_all
                a = loss_all * 1.0      

            loss = loss_surv  
            
            # 參數為T    
            if args.add_mse_loss_of_mae: 
                # 將該 MSE 的平均損失加入到總損失中
                mse_loss_of_mae /= iter
                mse_loss_of_mae = mse_loss_of_mae / 5  # 5
                loss += (mse_loss_of_mae)
                #print("mse_loss_of_mae:",mse_loss_of_mae)

            # 累加當前批次的損失到總損失
            total_loss += a.item()
            # 反向傳播
            loss.backward()
            
            # 若為第一個週期, 打印星號, 用於顯示進度
            if epoch == 0:
                print('*',end='')
            else:  
                optimizer.step()
        
            torch.cuda.empty_cache()
     

            loss_nn_all.append(loss.data.item())
            mse_loss_of_mae = 0.0
            loss_surv = 0.0
            iter = 0 
            loss_imgN = 0.0
            loss_imgA = 0.0
            loss_imgL = 0.0
            loss_cli = 0.0
            logits_all, logits_imgN, logits_imgA, logits_imgL, logits_cli = [], [], [], [], []
            diagnosis_batch_T = []

    # 當跑完所有 Train 的病患
    
    # 計算整體平均損失: 總損失 / 訓練數據的總量 * 批次大小
    # total_loss = total_loss/len(train_data)*batch_size
    #print("total_loss:",total_loss)
    total_loss = total_loss / len(train_data) * batch_size

    
    diagnosis_all = np.asarray(diagnosis_all)

    # 轉換列表為 NumPy 數組
    predicted_labels_all = np.concatenate(predicted_labels_all)
    predicted_labels_imgN = np.concatenate(predicted_labels_imgN)
    predicted_labels_imgA = np.concatenate(predicted_labels_imgA)
    predicted_labels_imgL = np.concatenate(predicted_labels_imgL)
    predicted_labels_cli = np.concatenate(predicted_labels_cli)

    # 計算效能指標
    accuracy_all = accuracy_score(diagnosis_all, predicted_labels_all)
    accuracy_imgN = accuracy_score(diagnosis_all, predicted_labels_imgN)
    accuracy_imgA = accuracy_score(diagnosis_all, predicted_labels_imgA)
    accuracy_imgL = accuracy_score(diagnosis_all, predicted_labels_imgL)
    accuracy_cli = accuracy_score(diagnosis_all, predicted_labels_cli)

    ## 新增: 計算混淆矩陣
    ## 轉換真實標籤為Numpy數組
    true_labels = np.array(true_labels)
    ## 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    ## 計算每個類別之評價指標
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    ## 根據混淆矩陣計算偽陰性（FP）、偽陽性（FN）、敏感性和特異性
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return total_loss, accuracy_all, accuracy_imgN, accuracy_imgA, accuracy_imgL, accuracy_cli,\
       # cm, precision, recall, f1, FP, FN, TP, TN, sensitivity, specificity
###===================###



###===================###
def main(args): 

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    val_losses, val_accuracies = [], []

    start_seed = args.start_seed
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr


    ###--- patients
    # 指定Excel文件的路徑
    excel_file_path = r'Excel文件路徑'
    # 使用Pandas讀取Excel文件，假設第一行是列名
    df = pd.read_excel(excel_file_path)
    # 提取所有病人編號，假設編號在'A'列
    patient_ids = df['編號'].tolist()

    ###--- patient_diagnosis_type
    df = pd.read_excel(r'Excel文件路徑') 
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

    # 提取 kf_label
    kf_label = [patient_diagnosis_type[id] for id in patient_ids]
    all_data=joblib.load(r'圖結構資料位置') # all_data_50.pkl、all_data_101.pkl

    all_fold_test_ac = []
    all_all_ac = []
    all_gnn_time = []
    all_each_model_time = []
    all_fold_each_model_ac = []


    # 第一步
    for seed in range(start_seed,start_seed+repeat_num):
 
        setup_seed(0)  
        seed_patients = []
        gnn_time = {}
        test_fold_ac = []

        val_fold_ac = []
        test_each_model_ac = {'imgN':{},'imgA':{},'imgL':{},'cli':{},
                           'imgNimgA':{},'imgNimgL':{},'imgNcli':{},
                           'imgAimgL':{},'imgAcli':{},
                           'imgLcli':{}}
        train_fold_ac=[]
        n_fold = 0
        all_accuracy = 0.0

        # 第二步
        kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state = seed)
        for train_index, test_index in kf.split(patient_ids, kf_label):
            fold_patients = []
            n_fold+=1
            print('fold: ',n_fold)

            # # 初始化該模型 
            if fusion_model == 'fusion_model_mae_2':
                model = fusion_model_mae_2(in_feats=1024,
                               n_hidden=args.n_hidden,
                               out_classes=args.out_classes,
                               dropout=drop_out_ratio,
                               train_type_num = len(args.train_use_type)
                                      ).to(device)
                model = model.to(device)  # 将模型移至 GPU

            optimizer=Adam(model.parameters(), lr = lr, weight_decay = 5e-4)  # weight_decay = 5e-4

            if args.if_fit_split:
                # 在這裡定義訓練、驗證和測試集
                train_data = [...]
                val_data = [...]
                test_data = [...]
            else: 
                # 從原始數據集根據訓練和測試索引分割數據集
                t_train_data = np.array(patient_ids)[train_index]
                t_l = []
                # 提取 訓練病患的病灶結果
                for x in t_train_data:
                    t_l.append(patient_diagnosis_type[x])
                train_data, val_data ,_ , _ = train_test_split(t_train_data,t_train_data,test_size=0.25,random_state=1,stratify=t_l)         
                test_data = np.array(patient_ids)[test_index]
                # train_data 佔60%, test_data 佔20%, val_data 佔20%

            print("訓練數據有", len(train_data), "筆"," ","驗證數據有", len(val_data), "筆"," ","測試數據有", len(test_data), "筆")
            
            fold_patients.append(train_data)
            fold_patients.append(val_data)
            fold_patients.append(test_data)
            seed_patients.append(fold_patients)
    
            best_val_ac = 0
            tmp_train_ac = 0
            t_model = None
            # 第三步
            for epoch in range(epochs):
                
                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=40, lr_gamma=args.adjust_lr_raito)    # lr_step=20
                
                train_loss,train_accuracy_all,train_accuracy_imgN,train_accuracy_imgA,train_accuracy_imgL,train_accuracy_cli = train_a_epoch(model, train_data, all_data, patient_diagnosis_type, batch_size, optimizer,epoch, format_of_coxloss, args)
                
                test_loss,test_accuracy_all,test_accuracy_imgN,test_accuracy_imgA,test_accuracy_imgL,test_accuracy_cli,t_cm,t_precision,t_recall,t_f1,t_FP,t_FN,t_TP,t_TN = prediction(all_data, model, test_data, patient_diagnosis_type, args)  
                val_loss,val_accuracy_all,val_accuracy_imgN,val_accuracy_imgA,val_accuracy_imgL,val_accuracy_cli,v_cm,v_precision,v_recall,v_f1,v_FP,v_FN,v_TP,v_TN = prediction(all_data, model, val_data, patient_diagnosis_type, args)
                if epoch == 0:
                    print("\n")
                print("epoch:{:2d}, train_loss:{:.4f}, train_ac:{:.4f}, val_loss:{:.4f}, val_ac:{:.6f}, test_loss:{:.4f}, test_ac:{:.4f}".format(epoch,train_loss,train_accuracy_all,val_loss,val_accuracy_all,test_loss,test_accuracy_all)) 
                print("epoch:{:2d}, train_accuracy_imgN:{:.4f}, train_accuracy_imgA:{:.4f}, train_accuracy_imgL:{:.4f}, train_accuracy_cli:{:.4f}".format(epoch,train_accuracy_imgN,train_accuracy_imgA,train_accuracy_imgL,train_accuracy_cli)) 

                if epoch == 0:
                    best_val_ac = val_accuracy_all
                    tmp_train_ac = train_accuracy_all
                    t_model = copy.deepcopy(model)
                    best_v_metrics = {
                        "v_cm": v_cm, "v_precision": v_precision, "v_recall": v_recall, "v_f1": v_f1, "v_FP": v_FP, "v_FN": v_FN, "v_TP": v_TP, "v_TN": v_TN,
                        "t_cm": t_cm, "t_precision": t_precision, "t_recall": t_recall, "t_f1": t_f1, "t_FP": t_FP, "t_FN": t_FN, "t_TP": t_TP, "t_TN": t_TN,
                    }
                    
                # 紀錄當前epoch最好的結果
                if val_accuracy_all > best_val_ac and epoch>=1 :
                    best_val_ac = val_accuracy_all
                    tmp_train_ac = train_accuracy_all
                    print("更新最佳驗證集準確率:{:.4f}".format(val_accuracy_all))
                    t_model = copy.deepcopy(model)
                    best_v_metrics = {
                        "v_cm": v_cm, "v_precision": v_precision, "v_recall": v_recall, "v_f1": v_f1, "v_FP": v_FP, "v_FN": v_FN, "v_TP": v_TP, "v_TN": v_TN,
                        "t_cm": t_cm, "t_precision": t_precision, "t_recall": t_recall, "t_f1": t_f1, "t_FP": t_FP, "t_FN": t_FN, "t_TP": t_TP, "t_TN": t_TN,
                    }
                    # 打印best_v_metrics中的所有值
                    for metric, value in best_v_metrics.items():
                        if isinstance(value, np.ndarray):
                            # 如果值是一个数组，打印数组的形状和一些元素示例
                            print(f"{metric}: shape={value.shape}")
                            print(value)  # 完整打印整个数组
                        else:
                            # 否则直接打印值
                            print(f"{metric}: {value}")
                
                # 在 epoch 循還
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy_all)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy_all)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy_all)
                
                # 繪製訓練圖表
                if epoch  == (epochs - 1) or epoch == 59 or epoch == 89 or epoch == 119 or epoch == 179 :
                    # 確保損失列表中的值都是標準數據類型
                    train_losses_cpu = [loss.item() if torch.is_tensor(loss) else loss for loss in train_losses]
                    test_losses_cpu = [loss.item() if torch.is_tensor(loss) else loss for loss in test_losses]
                    val_losses_cpu = [loss.item() if torch.is_tensor(loss) else loss for loss in val_losses]

                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label='Train Loss')
                    plt.plot(test_losses_cpu, label='Test Loss')
                    plt.plot(val_losses_cpu, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Loss Across Epochs')
                    plt.legend()
                    # 保存loss圖表
                    plt.savefig(f'存放loss圖表位置\.png')
                    plt.show()

                    # 绘制准确率图表
                    plt.figure(figsize=(10, 5))
                    plt.plot(train_accuracies, label='Train Accuracy')
                    plt.plot(test_accuracies, label='Test Accuracy')
                    plt.plot(val_accuracies, label='Validation Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Accuracy Across Epochs')
                    plt.legend()
                    # 保存Accuracy圖表
                    plt.savefig(f'存放Accuracy圖表位置\.png')
                    plt.show()
                
            t_model.eval() 

            # 使用最佳模型
            t_test_loss, t_test_accuracy_all,_,_,_,_ = prediction(all_data,t_model,test_data,patient_diagnosis_type,args)  
            # 儲存每個週期最佳模型的測試集 ac
            test_fold_ac.append(t_test_accuracy_all)
            # 儲存每個週期最佳模型的驗證集 ac
            val_fold_ac.append(best_val_ac)
            # 儲存每個週期最佳模型的訓練集 ac
            train_fold_ac.append(tmp_train_ac)

            one_model_res = [{},{},{},{}]
            two_model_res = [{},{},{},{},{},{}]
            fold_fusion_test_ci = {}
            diagnosis_test = []

            patient_one_accuracy = {}
            patient_two_accuracy = {}
            patient_all_accuracy = {}

            # 使用全局变量
            for id in test_data:
                # 确保键存在
                if id not in patient_all_accuracy:
                    patient_all_accuracy[id] = {}  # 初始化為空字典
                if id not in patient_two_accuracy:
                    patient_two_accuracy[id] = {}  # 初始化為空字典
                if id not in patient_one_accuracy:
                    patient_one_accuracy[id] = {}  # 初始化為空字典   
###===================###


###===================###
# 在 Jupyter Notebook 中設定參數
args = {
    'imgN_cox_loss_factor': 5,
    'imgA_cox_loss_factor': 5,
    'imgL_cox_loss_factor': 5,
    'cli_cox_loss_factor': 5,
    'train_use_type': ['imgN', 'imgA', 'imgL', 'cli'],
    'format_of_coxloss': "multi",
    'add_mse_loss_of_mae': True,
    'mse_loss_of_mae_factor': 5,
    'start_seed': 0,
    'repeat_num': 1,   # 調 0 讓他直接結束
    'fusion_model': "fusion_model_mae_2",
    'drop_out_ratio': 0.3,
    'lr': 0.0001,   #0.0001
    'epochs': 180,    # 60
    'batch_size': 8,   #8
    'n_hidden': 512,
    'out_classes': 512,
    'mix': True,
    'if_adjust_lr': True,
    'adjust_lr_raito': 0.8, # 0.5
    'if_fit_split': True,
    'details': ''
}

# 將字典轉換為一個簡單的 class 來模擬 argparse 返回的 Namespace
class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = Args(**args)
###===================###

###===================###
# 調用 main 函數
main(args)
###===================###

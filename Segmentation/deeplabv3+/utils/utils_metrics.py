import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# inputs是模型預測的輸出　（n,c,h,w）批量大小、類別數
# target是真實標籤
def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    # 計算真陽性(TP, 正確預測的正類數)
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    # 計算假陽性(FP, 錯誤預測為正類的負類數)
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    # 計算假陰性(FN, 錯誤預測為負類的正類數)
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  


# ex:類別編號分別為狗(0)、貓(1)、兔子(2)、背景(3)
# [[5, 2, 0, 1],  # 狗   : 第一行表示真實標籤為狗的像素，其中5個正確預測為狗，2個錯誤預測為貓，1個錯誤預測為背景。
#  [1, 7, 1, 0],  # 貓   : 第二行表示真實標籤為貓的像素，其中7個正確預測為貓，1個錯誤預測為狗，1個錯誤預測為兔子。
#  [0, 2, 8, 0],  # 兔子 : 第三行表示真實標籤為兔子的像素，其中8個正確預測為兔子，2個錯誤預測為貓。
#  [0, 0, 0, 9]]  # 背景 : 第四行表示真實標籤為背景的像素，全部9個正確預測為背景。
#  
#  IoU = TP/(TP+FP+FN)
#  狗的IoU = 5 / (5 + 1 + 3) = 5/9
#  貓的IoU = 7 / (7 + 4 + 2) = 7/13
#  兔子的IoU = 8 / (8 + 1 + 2) = 8/11
#  背景的IoU = 9 / (9 + 1 + 0) = 9/10
###################################################################################################################
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

# ex:類別編號分別為狗(0)、貓(1)、背景(2)
# [[80, 20,  0],  # 狗
#  [10, 70, 20],  # 貓
#  [ 0, 30, 70]]  # 背景
#
# 計算召回率，即模型正確預測為該類別的像素占所有真實為該類別像素的比例
#  PA_Recall = TP/(TP+FN)
#  TP（True Positives）是對角線上的值，代表每個類別被正確預測的像素數。
#  FN（False Negatives）是該行其餘元素的和，代表該類別被錯誤預測為其他類別的像素數。
#
#  對於狗:TP= 80
#         FN= 20 + 0 = 20
#         PA_Recall  = 80 / (80 + 20) = 0.8
#  對於貓:TP= 70
#         FN= 10 + 20 = 30
#         PA_Recall  = 70 / (70 + 30) = 0.7
#  對於狗:TP= 70
#         FN= 0 + 30 = 30
#         PA_Recall  = 70 / (70 + 30) = 0.7
def per_class_PA_Recall(hist): 
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

# ex:類別編號分別為狗(0)、貓(1)、背景(2)
# [[80, 5,  15],  # 狗
#  [20, 75, 5],   # 貓
#  [0,  20, 80]]  # 背景
#
# 計算精準度（Precision），即模型預測為該類別的像素中，真正正確屬於該類別的像素占的比例
# Precision = TP/(TP+FP)
# TP（True Positives）是對角線上的值，表示每個類別被正確預測的像素數量。
# FP（False Positives）是該列其餘元素的和，代表被錯誤預測為該類別的像素數量。
#
#  對於狗:TP= 80
#         FP= 20 + 0 = 20
#         Precision  = 80 / (80 + 20) = 0.8
#  對於貓:TP= 75
#         FP= 5 + 20 = 25
#         Precision  = 75 / (75 + 25) = 0.75
#  對於狗:TP= 80
#         FP= 15 + 5 = 20
#         Precision  = 80 / (80 + 20) = 0.8
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

# ex:類別編號分別為狗(0)、貓(1)、背景(2)
# [[2, 1, 0],  # 狗
#  [0, 1, 1],  # 貓
#  [0, 0, 3]]  # 背景
#
# 
def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

# 平均交并比（mean Intersection over Union, mIoU）、每類別像素精度（mean Pixel Accuracy, mPA）和總體精度（Accuracy）
# 真實標籤的目錄gt_dir、預測結果的目錄pred_dir、圖像名稱列表png_name_list、
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    name_classes = ["Background", "AW", "Puncation", "Mosaic", "Atypical"]
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #   用於統計每個類別被正確和錯誤預測的次數
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    #   獲得驗證集標籤路徑列表，方便直接讀取
    #   獲得驗證集圖像分割結果路徑列表，方便直接讀取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   遍歷每一對圖像和標籤
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            """
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            """
            continue

        #------------------------------------------------#
        #   对一张图片计算5×5的hist矩阵，并累加
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    #------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)         # 若num_classes=5，則IoUs為一個數組包含5個類別各自IoU直。
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #   逐类别输出一下mIoU值
    #   遍歷每個類別，並分別打印出該類別的IoU、Recall、Precision
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #   打印出平均指標:mIoU、mPA、整體準確率        
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, int), IoUs, PA_Recall, Precision # np.int

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            
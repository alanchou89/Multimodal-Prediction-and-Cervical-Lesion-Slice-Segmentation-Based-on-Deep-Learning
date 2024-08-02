import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv,LayerNorm
from mae_utils import get_sinusoid_encoding_table,Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 此函數用於遍歷神經網路模型的所有層，並重新初始化每一層的權重和偏差。
# 這在重新訓練模型之前經常使用，以確保模型從一個乾淨的狀態開始。
def reset(nn):
    def _reset(item): 
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

# 實現了一種全局注意力機制，通過 gate_nn 來學習特徵的重要性，並根據這些重要性對特徵進行加權和聚合。
# 這在處理圖數據和其他類型的序列數據時非常有用
class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out,gate


    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    
# 用於初始化權重，使其遵循截斷的正態分佈。  
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False,train_type_num=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        num_patches = train_type_num

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,train_type_num=2,
                 ):
        super().__init__()
        self.num_classes = num_classes
#         assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=512, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.3,
                 drop_path_rate=0.3, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 train_type_num=2,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=3,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            train_type_num=train_type_num)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#         self.mask_token = torch.zeros(1, 1, decoder_embed_dim).to(device)
        

        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        B, N, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, 0) # [B, N_mask, 3 * 16 * 16]

        tmp_x = torch.zeros_like(x).to(device)
        Mask_n = 0
        Truth_n = 0
        for i,flag in enumerate(mask[0][0]):
            if flag:  
                tmp_x[:,i] = x[:,pos_emd_vis.shape[1]+Mask_n]
                Mask_n += 1
            else:
                tmp_x[:,i] = x[:,Truth_n]
                Truth_n += 1
        return tmp_x

# 建立一個簡單的多層感知器（MLP）結構: 用於提取特徵和進行非線性轉換
def Mix_mlp(dim1):
    
    return nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim1))
# 用於混合不同特徵，進行更有效的特徵提取。
class MixerBlock(nn.Module):
    # 接收兩個參數 dim1 和 dim2，用於定義內部層的維度
    def __init__(self,dim1,dim2):
        super(MixerBlock,self).__init__() 
        
        self.norm = LayerNorm(dim2)
        self.mix_mip_1 = Mix_mlp(dim1) # 4
        self.mix_mip_2 = Mix_mlp(dim2) # 512
        
    def forward(self,x): 
        
        y = self.norm(x)        # [4, 512]
        y = y.transpose(0,1)    # [512, 4]
        y = self.mix_mip_1(y)
        y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        x = x + self.mix_mip_2(y)
        
#         y = self.norm(x)
#         y = y.transpose(0,1)
#         y = self.mix_mip_1(y)
#         y = y.transpose(0,1)
#         x = self.norm(y)
        return x

# 建立多層感知機（MLP）塊，其結構包括線性層、激活函數和丟棄層:用於提取特徵和增加模型的表達能力
def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

def GNN_relu_Block(dim2, dropout=0.3): # dropout=0.3
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
#             GATConv(in_channels=dim1,out_channels=dim2),
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))

class fusion_model_mae_2(nn.Module):
    def __init__(self,in_feats,n_hidden,out_classes,dropout=0.3,train_type_num=2):  # dropout=0.3
        super(fusion_model_mae_2,self).__init__() 

        # 用於 imgN 模態的圖卷積層
        self.imgN_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)     
        # ReLU激活和LayerNorm正規化的圖神經網絡層
        self.imgN_relu_2 = GNN_relu_Block(out_classes)  
        # 用於 imgA 模態的圖卷積層
        self.imgA_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)     
        # ReLU激活和LayerNorm正規化的圖神經網絡層
        self.imgA_relu_2 = GNN_relu_Block(out_classes) 
        # 用於 imgL 模態的圖卷積層
        self.imgL_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)     
        # ReLU激活和LayerNorm正規化的圖神經網絡層
        self.imgL_relu_2 = GNN_relu_Block(out_classes)     
        # 用於 cli 模態的圖卷積層
        self.cli_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes) 
        # ReLU激活和LayerNorm正規化的圖神經網絡層        
        self.cli_relu_2 = GNN_relu_Block(out_classes) 

        # 因我的臨床數據不適合進行圖卷積 故:
        self.fc_cli_1 = nn.Linear(1024, out_classes)
        self.fc_cli_2 = nn.Linear(out_classes, out_classes)

#         TransformerConv

        # 以下解析
        # 生成特徵的加權分數：att_net_img 是一個序列模組，首先通過一個線性層將特徵維度從 out_classes 減少到 out_classes//4，
        #                   然後應用非線性激活函數（ReLU），最後再經過另一個線性層將維度從 out_classes//4 變為 1。這個流程為每個特徵向量生成一個加權分數（即重要性評分）。
        # 執行加權平均：my_GlobalAttention 模組接收 att_net_img 的輸出作為門控值（即加權分數），並對特徵向量進行加權平均。
        #              這使得模型能夠專注於那些被認為更重要的特徵，從而提高模型的表達能力和準確性。

        # 生成一個特徵的加權分數 (純量): 功能是對每個輸入特徵計算一個分數（或稱重要性）
        att_net_imgN = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        # 將分數執行加權平均，這樣模型就可以專注於那些更重要的特徵。
        self.mpool_imgN = my_GlobalAttention(att_net_imgN)
        att_net_imgA = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_imgA = my_GlobalAttention(att_net_imgA)
        att_net_imgL = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_imgL = my_GlobalAttention(att_net_imgL)        
        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli = my_GlobalAttention(att_net_cli)

        att_net_imgN_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_imgN_2 = my_GlobalAttention(att_net_imgN_2)
        att_net_imgA_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_imgA_2 = my_GlobalAttention(att_net_imgA_2)
        att_net_imgL_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_imgL_2 = my_GlobalAttention(att_net_imgL_2)       
        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)
        
        
        self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1,decoder_depth=1,train_type_num=train_type_num)
        # 用於融合不同特徵或輸入數據。  數據類型數量、輸出維度
        self.mix = MixerBlock(train_type_num, out_classes)
        
        # 全連接層
        self.lin1_imgN = torch.nn.Linear(out_classes,out_classes//4)             # 512 128
        self.lin2_imgN = torch.nn.Linear(out_classes//4,out_classes//4//4)       # 128 32  
        self.lin3_imgN = torch.nn.Linear(out_classes//4//4,out_classes//4//4//4) # 32  8
        self.lin1_imgA = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_imgA = torch.nn.Linear(out_classes//4,out_classes//4//4)
        self.lin3_imgA = torch.nn.Linear(out_classes//4//4,out_classes//4//4//4) 
        self.lin1_imgL = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_imgL = torch.nn.Linear(out_classes//4,out_classes//4//4)
        self.lin3_imgL = torch.nn.Linear(out_classes//4//4,out_classes//4//4//4)
        self.lin1_cli = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_cli = torch.nn.Linear(out_classes//4,out_classes//4//4)   
        self.lin3_cli = torch.nn.Linear(out_classes//4//4,out_classes//4//4//4)      
        # 正規化層
        self.norm1_imgN = LayerNorm(out_classes//4)
        self.norm1_imgA = LayerNorm(out_classes//4)
        self.norm1_imgL = LayerNorm(out_classes//4)
        self.norm1_cli = LayerNorm(out_classes//4)
        self.norm2_imgN = LayerNorm(out_classes//4//4)
        self.norm2_imgA = LayerNorm(out_classes//4//4)
        self.norm2_imgL = LayerNorm(out_classes//4//4)
        self.norm2_cli = LayerNorm(out_classes//4//4)
        self.norm3_imgN = LayerNorm(out_classes//4//4//4)
        self.norm3_imgA = LayerNorm(out_classes//4//4//4)
        self.norm3_imgL = LayerNorm(out_classes//4//4//4)
        self.norm3_cli = LayerNorm(out_classes//4//4//4)
        # ReLU、Dropout層
        self.relu = torch.nn.ReLU() 
        self.dropout=nn.Dropout(p=dropout)
        
        # 新增一個全連接層用於類別預測
        self.classifier = nn.Linear(out_classes//4//4//4, 4) # 假設有 4 種病灶類型
        # 新增模態特定的分類器
        self.classifier_imgN = nn.Linear(out_classes//4//4//4, 4)  # 假設有 4 種病灶類型
        self.classifier_imgA = nn.Linear(out_classes//4//4//4, 4)
        self.classifier_imgL = nn.Linear(out_classes//4//4//4, 4)
        self.classifier_cli  = nn.Linear(out_classes//4//4//4, 4)


    def forward(self,all_thing,train_use_type=None,use_type=None,in_mask=[],mix=False):
        # 奇怪
        img_logits = None
        imgN_logits = None
        imgA_logits = None
        class_logits_imgA = None
        class_logits_imgL = None
        cli_logits = None
        mae_x = None

        # 沒有提供遮罩，則創建一個所有元素為 False 的遮罩
        if len(in_mask) == 0:
            mask = np.array([[[False]*len(train_use_type)]])
        else: # 直接使用該遮罩
            mask = in_mask
        

        # 當前使用數據類型
        data_type = use_type
        # 提取數據
        x_imgN = all_thing.x_imgN   #[4, 1024]     [16, 1024]
        x_imgA = all_thing.x_imgA   #[4, 1024]     [16, 1024]
        x_imgL = all_thing.x_imgL   #[4, 1024]     [16, 1024]
        x_cli = all_thing.x_cli     #[1, 1024]     [4, 1024]
        # 提取病患ID
        data_id = all_thing.data_id
        # 提取數據的邊緣索引
        edge_index_imgN = all_thing.edge_index_imageN   #[2, 8]    #[2, 84]
        edge_index_imgA = all_thing.edge_index_imageA   #[2, 8]    #[2, 84]
        edge_index_imgL = all_thing.edge_index_imageL   #[2, 8]    #[2, 84]
        edge_index_cli = all_thing.edge_index_cli       #[2, 90]   #[2, 12]

        
        save_fea = {}
        fea_dict = {}
        num_imgN = len(x_imgN)
        num_imgA = len(x_imgA)
        num_imgL = len(x_imgL)
        num_cli = len(x_cli)      
               
            
        att_2 = []
        pool_x = torch.empty((0)).to(device)
        # 圖卷積、全局注意力池化、特徵合併(pool_x)
        if 'imgN' in data_type:
            # 通過對應的圖卷積層處理數據
            x_imgN = self.imgN_gnn_2(x_imgN, edge_index_imgN)
            # ReLU、LayerNorm正規化 
            x_imgN = self.imgN_relu_2(x_imgN) 
            # 創建一個全零的批次向量  
            batch = torch.zeros(len(x_imgN),dtype=torch.long).to(device)
            # 通過對應的全局注意力池化層進行特徵提取和注意力分數計算
            pool_x_imgN, att_imgN_2 = self.mpool_imgN(x_imgN, batch)
            # 提取的注意力分數加到相應的列表
            att_2.append(att_imgN_2)
            # 將處理後的特徵串接到一個統一的特徵向量
            pool_x = torch.cat((pool_x,pool_x_imgN),0)
        if 'imgA' in data_type:
            # 通過對應的圖卷積層處理數據
            x_imgA = self.imgA_gnn_2(x_imgA, edge_index_imgA)
            # ReLU、LayerNorm正規化 
            x_imgA = self.imgA_relu_2(x_imgA) 
            # 創建一個全零的批次向量  
            batch = torch.zeros(len(x_imgA),dtype=torch.long).to(device)
            # 通過對應的全局注意力池化層進行特徵提取和注意力分數計算
            pool_x_imgA, att_imgA_2 = self.mpool_imgA(x_imgA, batch)
            # 提取的注意力分數加到相應的列表
            att_2.append(att_imgA_2)
            # 將處理後的特徵串接到一個統一的特徵向量
            pool_x = torch.cat((pool_x,pool_x_imgA),0)
        if 'imgL' in data_type:
            # 通過對應的圖卷積層處理數據
            x_imgL = self.imgL_gnn_2(x_imgL, edge_index_imgL)
            # ReLU、LayerNorm正規化 
            x_imgL = self.imgL_relu_2(x_imgL) 
            # 創建一個全零的批次向量  
            batch = torch.zeros(len(x_imgL),dtype=torch.long).to(device)
            # 通過對應的全局注意力池化層進行特徵提取和注意力分數計算
            pool_x_imgL,att_imgL_2 = self.mpool_imgL(x_imgL, batch)
            # 提取的注意力分數加到相應的列表
            att_2.append(att_imgL_2)
            # 將處理後的特徵串接到一個統一的特徵向量
            pool_x = torch.cat((pool_x,pool_x_imgL),0)       
        if 'cli' in data_type:
            #
            x_cli = self.cli_gnn_2(x_cli,edge_index_cli) 
            x_cli = self.cli_relu_2(x_cli)   
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_2 = self.mpool_cli(x_cli,batch)
            att_2.append(att_cli_2)
            pool_x = torch.cat((pool_x,pool_x_cli),0)
            
        # pool_x 為綜合特徵向量
        # 將組合後的特徵向量存入 fea_dict 字典中，鍵名為 'mae_labels
        fea_dict['mae_labels'] = pool_x

        ###=========================================================================###
        # 對合併的特徵向量pool_x進行自編碼器（MAE）處理，以生成新的特徵表示mae_x，並移除多餘維度。
        # 如果啟用了特徵混和（mix），則對mae_x進行進一步的混和處理
        # 將mae_x中相應的部分加到原始的模態特徵上
        #
        # 是否有多於一種模態
        if len(train_use_type)>1:
            # 全模態都有
            if use_type == train_use_type:
                # 綜合特徵向量 進行自編碼器（MAE）處理, 並移除多餘維度 ex:(1, N)->(N,)
                mae_x = self.mae(pool_x,mask).squeeze(0)
                #print("mae_x size:", mae_x.size())    #### [4,512]
                fea_dict['mae_out'] = mae_x
            else: # 加的
                k=0
                tmp_x = torch.zeros((len(train_use_type),pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type),dtype=bool)
                for i,type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k+=1
                        mask[i] = False
                mask = np.expand_dims(mask,0)
                mask = np.expand_dims(mask,0)
                if k==0:
                    mask = np.array([[[False]*len(train_use_type)]])
                mae_x = self.mae(tmp_x,mask).squeeze(0)
                fea_dict['mae_out'] = mae_x   

            # MAE處理後的特徵從GPU移至CPU，轉換為NumPy數組
            save_fea['after_mae'] = mae_x.cpu().detach().numpy() 
            # 啟用特徵混和
            if mix:
                # 對MAE處理後的特徵進行混合處理
                mae_x = self.mix(mae_x)
                # 混合後的特徵同樣從GPU移至CPU並轉換為NumPy數組
                save_fea['after_mix'] = mae_x.cpu().detach().numpy() 

            k=0
            # 檢查數據是否在訓練和使用的類型中。如果是，則對應的特徵會被更新
            if 'imgN' in train_use_type and 'imgN' in use_type:
                x_imgN = x_imgN + mae_x[train_use_type.index('imgN')] 
                k+=1
            if 'imgA' in train_use_type and 'imgA' in use_type:
                x_imgA = x_imgA + mae_x[train_use_type.index('imgA')] 
                k+=1
            if 'imgL' in train_use_type and 'imgL' in use_type:
                x_imgL = x_imgL + mae_x[train_use_type.index('imgL')] 
                k+=1
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]  
                k+=1
        ###=========================================================================###      
        # 再次使用池化層,提取更加豐富和深入的特徵
        att_3 = []
        pool_x = torch.empty((0)).to(device)
        
        if 'imgN' in data_type:
            batch = torch.zeros(len(x_imgN),dtype=torch.long).to(device)
            pool_x_imgN,att_imgN_3 = self.mpool_imgN_2(x_imgN,batch)
            att_3.append(att_imgN_3)
            pool_x = torch.cat((pool_x,pool_x_imgN),0)
        if 'imgA' in data_type:
            batch = torch.zeros(len(x_imgA),dtype=torch.long).to(device)
            pool_x_imgA,att_imgA_3 = self.mpool_imgA_2(x_imgA,batch)
            att_3.append(att_imgA_3)
            pool_x = torch.cat((pool_x,pool_x_imgA),0)
        if 'imgL' in data_type:
            batch = torch.zeros(len(x_imgL),dtype=torch.long).to(device)
            pool_x_imgL,att_imgL_3 = self.mpool_imgL_2(x_imgL,batch)
            att_3.append(att_imgL_3)
            pool_x = torch.cat((pool_x,pool_x_imgL),0)
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_3 = self.mpool_cli_2(x_cli,batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x,pool_x_cli),0) 
            
        # 综合了所有可用模态特征的张量
        x = pool_x
        # 進行 L2 正規化，使得每個樣本的特徵向量長度為 1 : 助於改善訓練穩定性和模型性能
        x = F.normalize(x, dim=1) # 打印 x 的尺寸 [4,512]
        fea = x
        ###=========================================================================###
        k=0
        if 'imgN' in data_type:
            # 提取對應的特徵並保存到字典中
            fea_dict['imgN'] = fea[k]
            k+=1
        if 'imgA' in data_type:
            # 提取對應的特徵並保存到字典中
            fea_dict['imgA'] = fea[k]
            k+=1
        if 'imgL' in data_type:
            # 提取對應的特徵並保存到字典中
            fea_dict['imgL'] = fea[k]
            k+=1
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
            k+=1

        ###  用於進一步處理從不同模態（如圖像、RNA、臨床數據）獲得的特徵。
        k=0
        # 儲存處理後的特徵
        multi_x = torch.empty((0)).to(device)

        if 'imgN' in data_type:
            x_imgN = self.lin1_imgN(x[k])  # 512->128
            # ReLU
            x_imgN = self.relu(x_imgN)
            # 正則化
            x_imgN = self.norm1_imgN(x_imgN)
            # dropout
            x_imgN = self.dropout(x_imgN)
            x_imgN = self.lin2_imgN(x_imgN) # 128->32
            x_imgN = self.relu(x_imgN)      # 之後可以考慮刪減
            x_imgN = self.norm2_imgN(x_imgN) # 之後可以考慮刪減
            x_imgN = self.dropout(x_imgN)   # 之後可以考慮刪減
            x_imgN = self.lin3_imgN(x_imgN) # 32->8
            # img_logits = self.classifier_imgN(x_imgN)  # 8-> 4
            imgN_logits = self.classifier_imgN(x_imgN)  # 8-> 4
            x_imgN = x_imgN.unsqueeze(0)
            multi_x = torch.cat((multi_x,x_imgN),0)  #[1,32]
            k+=1
            
        if 'imgA' in data_type:
            x_imgA = self.lin1_imgA(x[k])  # 512->128
            x_imgA = self.relu(x_imgA)
            x_imgA = self.norm1_imgA(x_imgA)
            x_imgA = self.dropout(x_imgA)
            x_imgA = self.lin2_imgA(x_imgA) # 128->32
            x_imgA = self.relu(x_imgA)
            x_imgA = self.norm2_imgA(x_imgA)
            x_imgA = self.dropout(x_imgA)
            x_imgA = self.lin3_imgA(x_imgA) # 32->8
            # img_logits = self.classifier_imgA(x_imgA) 
            imgA_logits = self.classifier_imgA(x_imgA) 
            x_imgA = x_imgA.unsqueeze(0) #[1,32]
            multi_x = torch.cat((multi_x,x_imgA),0)  #[2,32]
            k+=1
            
        if 'imgL' in data_type:
            x_imgL = self.lin1_imgL(x[k])  # 512->128
            x_imgL = self.relu(x_imgL)
            x_imgL = self.norm1_imgL(x_imgL)
            x_imgL = self.dropout(x_imgL)
            x_imgL = self.lin2_imgL(x_imgL) # 128->32
            x_imgL = self.relu(x_imgL)
            x_imgL = self.norm2_imgL(x_imgL)
            x_imgL = self.dropout(x_imgL)
            x_imgL = self.lin3_imgL(x_imgL) # 32->8
            # img_logits = self.classifier_imgL(x_imgL) 
            imgL_logits = self.classifier_imgL(x_imgL) 
            x_imgL = x_imgL.unsqueeze(0) #[1,32]
            multi_x = torch.cat((multi_x,x_imgL),0)  #[3,32]
            k+=1
            
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])  # 512->128
            x_cli = self.relu(x_cli)
            x_cli = self.norm1_cli(x_cli)
            x_cli = self.dropout(x_cli)
            x_cli = self.lin2_cli(x_cli) # 128->32
            x_cli = self.relu(x_cli)
            x_cli = self.norm2_cli(x_cli)
            x_cli = self.dropout(x_cli)
            x_cli = self.lin3_cli(x_cli) # 32->8
            cli_logits = self.classifier_cli(x_cli) # 8->4
            x_cli = x_cli.unsqueeze(0) #[1,32]
            multi_x = torch.cat((multi_x,x_cli),0)  #[4,32]
            k+=1
        #print("multi_x size:", multi_x.size())  # 打印 multi_x 的尺寸  
        # 計算 multi_x 中所有特徵的平均值，生成一個統一的特徵表示 one_x。這有助於獲得一個綜合所有模態的代表性特徵。
        one_x = torch.mean(multi_x,dim=0)
        #print("one_x size:", one_x.size())  # 打印 one_x 的尺寸
        # 使用 'one_x' 進行類別預測
        all_logits = self.classifier(one_x)
        

        # 返回:
        # (組合後的平均特徵 one_x, 所有模態的特徵集合)
        # 之前處理階段中保存的特徵
        # (從注意力機制中獲得的注意力分數)
        # 包含不同模態特定特徵的字典
        return (
            (one_x, multi_x),
            save_fea,
            (att_2, att_3),
            fea_dict,
            all_logits,
            imgN_logits,
            imgL_logits
        )












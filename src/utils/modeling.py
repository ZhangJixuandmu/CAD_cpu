# 更改环境变量为镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['USE_SAFETENSORS'] = '0' 

import torch
import torch.nn as nn
from transformers import AutoModel, BertModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel





# BERTweet-LSTM
class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select,dropout):

        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        #tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
        #model = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
        #tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
        #model = AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")

        # self.bert = AutoModel.from_pretrained("bertweet-base")
        self.bert = AutoModel.from_pretrained("vinai/bertweet-base")

        # 添加提示
        try:
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            model = AutoModel.from_pretrained("vinai/bertweet-base")
            print("模型加载成功!")
        except Exception as e:
            print(f"加载失败: {e}")

        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear2 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.out2 = nn.Linear(self.bert.config.hidden_size, 2)

        #检查LSTM输入大小
        hidden_size = self.bert.config.hidden_size
        print(f"LSTM hidden size: {hidden_size}")
        # LSTM 输入大小应该是 hidden_size * 2
        self.lstm = nn.LSTM(self.bert.config.hidden_size*2, self.bert.config.hidden_size,bidirectional=True)
        # self.MultiAttention = MultiHeadedAttention()
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len, x_input_ids2):

        last_hidden = self.bert(input_ids=x_input_ids, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        last_hidden2 = self.bert(input_ids=x_input_ids2, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        ccc = self.bert.config.hidden_size*2
        # print(ccc)
        query = last_hidden[0][:,0]
        query2 = last_hidden2[0][:,0]
        query = self.dropout(query)
        query2 = self.dropout(query2)

        context_vec = torch.cat((query, query2), dim=1)
        # 添加调试信息
        print(f"Context vec shape: {context_vec.shape}")
        print(f"LSTM input size: {self.bert.config.hidden_size*2}")
        
        # 确保输入形状正确
        # LSTM 期望输入形状: (seq_len, batch, input_size) 或 (batch, seq_len, input_size)
        # 您需要调整 context_vec 的形状
        context_vec = context_vec.unsqueeze(0)  # 添加序列长度维度
        print(f"Reshaped context vec: {context_vec.shape}")
        # 确保 LSTM 的输入形状正确
        out1, h_n = self.lstm(context_vec)  # lstm层

        aaa = self.linear(query)
        linear = self.relu(aaa)
        out = self.out(linear)
        # linear2 = self.relu(self.linear2(context_vec))

        bbb = self.linear2(out1)
        linear2 = self.relu(bbb)
        out2 = self.out2(linear2)

        feature1 = out2.unsqueeze(1)
        feature1 = F.normalize(feature1, dim=2)

        feature2 = out2.unsqueeze(1)
        feature2 = F.normalize(feature2, dim=2)
        # 添加调试信息
        print(f"Output1 shape: {out1.shape}")
        print(f"Output2 shape: {out2.shape}")
        print(f"Feature1 shape: {feature1.shape if feature1 is not None else 'None'}")
        print(f"Feature2 shape: {feature2.shape if feature2 is not None else 'None'}")
    

        # 确保输出形状正确
        # out 应该具有形状 [batch_size, num_labels]
        # out2 应该具有形状 [batch_size, 2]
        
        # 如果 out2 的形状不正确，可能需要调整
        if out2.dim() > 2:
            out2 = out2.squeeze()  # 移除不必要的维度
        
        # 如果批次大小不匹配，可能需要复制或调整
        if out2.size(0) != x_input_ids.size(0):
            # 这是一个临时修复，您需要根据实际情况调整
            out2 = out2.expand(x_input_ids.size(0), -1)
    
    
        return out, out2, feature1, feature2
# 更改环境变量为镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['USE_SAFETENSORS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 设置为离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'   # 数据集也离线

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import utils.preprocessing as pp
import utils.data_helper as dh
# from transformers import AdamW
from torch.optim import AdamW
from utils import modeling, model_eval




def run_classifier():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="trump_hillary")# "trump_hillary"--trump_ted--hillary_bernie
    parser.add_argument("--model_select", type=str, default="BERTweet")
    parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
    parser.add_argument("--train_mode", type=str, default="unified")
    parser.add_argument("--lr", type=float, default=2e-5)#2e-5
    parser.add_argument("--batch_size", type=int, default=16)# adhoc :16
    parser.add_argument("--epochs", type=int, default=20)#20  adhoc: 10
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)# 0.5
    args = parser.parse_args()

    random_seeds = [2,3,4,5,6,7]
    target_word_pair = [args.input_target]
    model_select = args.model_select
    col = args.col
    train_mode = args.train_mode
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    alpha = args.alpha

    # 获取设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #Creating Normalization Dictionary
    with open("src/noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("src/emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1,**data2}

    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []
        for seed in random_seeds:    
            print("current random seed: ", seed)

            # filename1 = '../data/raw_train_all.csv'
            # filename2 = '../data/raw_val_all.csv'
            # filename3 = '../data/raw_test_all.csv'
            filename1 = 'data/raw_train_trump_hillary.csv'
            filename2 = 'data/raw_val_trump_hillary.csv'
            filename3 = 'data/raw_test_trump_hillary.csv'

            x_train,y_train,x_train_target,y_train2,x_train_target2 = \
                                        pp.clean_all(filename1,'Stance1',normalization_dict)
            x_val,y_val,x_val_target,y_val2,x_val_target2 = \
                                        pp.clean_all(filename2,'Stance1',normalization_dict)
            x_test,y_test,x_test_target,y_test2,x_test_target2 = \
                                        pp.clean_all(filename3,'Stance1',normalization_dict)
                
            num_labels = len(set(y_train))
            # print(x_train_target[0])
            x_train_all = [x_train,y_train,x_train_target,y_train2,x_train_target2]
            x_val_all = [x_val,y_val,x_val_target,y_val2,x_val_target2]
            x_test_all = [x_test,y_test,x_test_target,y_test2,x_test_target2]
            
            # set up the random seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 

            # prepare for model
            x_train_all,x_val_all,x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,\
                                        target_word_pair[target_index],model_select)
            # print(x_test_all[0][0])
            x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, \
                                        trainloader, x_train_input_ids2, y_train2 = \
                                        dh.data_load(x_train_all, batch_size, train_mode)
            x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader, \
                                        x_val_input_ids2, y_val2 = \
                                        dh.data_load(x_val_all, batch_size, train_mode)                            
            x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len, testloader, \
                                        x_test_input_ids2, y_test2 = \
                                        dh.data_load(x_test_all, batch_size, train_mode)

            # model = modeling.stance_classifier(num_labels,model_select,dropout).cuda()
            model = modeling.stance_classifier(num_labels,model_select,dropout).to(device)

            for n,p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
                ]
            
            # loss_function1 = nn.CrossEntropyLoss(reduction='sum')
            # loss_function2 = Stance_loss(temperature=0.07).to('cuda')#0.07
            # optimizer = AdamW(optimizer_grouped_parameters)
            loss_function1 = nn.CrossEntropyLoss(reduction='sum')
            print(f"Loss function 1: {loss_function1}")
            # #添加形状检查
            # print(f"Output2 shape before loss: {output2.shape}")
            # print(f"Target2 shape before loss: {target2.shape}")
            loss_function2 = Stance_loss(temperature=0.07).to(device)#0.07
            optimizer = AdamW(optimizer_grouped_parameters)


            sum_loss = []
            sum_val = []
            val_f1_average = []

            test_f1_average = [[] for i in range(2)]


            for epoch in range(0, total_epoch):
                print('Epoch:', epoch)
                train_loss, valid_loss = [], []
                model.train()
                print("Starting training loop...")
                for i, (input_ids, seg_ids, atten_masks, target, length, input_ids2, target2) in enumerate(trainloader):
                    optimizer.zero_grad()
                    print("Before forward pass")

                    output1, output2, feature1, feature2 = model(input_ids, seg_ids, atten_masks, length, input_ids2)
                    print("After forward pass")

                    # 添加调试信息
                    print(f"Output1 shape: {output1.shape}")
                    print(f"Output2 shape: {output2.shape}")
                    print(f"Target shape: {target.shape}")
                    print(f"Target2 shape: {target2.shape}")
                    print(f"Feature1 shape: {feature1.shape}")
                    print(f"Feature2 shape: {feature2.shape}")

                    loss1 = loss_function1(output1, target)
                    loss2 = loss_function1(output2, target2)

                    # 重塑特征匹配标签形状
                    feature1 = feature1.view(-1,2)
                    feature2 = feature2.view(-1,2)
                    target = target.long()
                    target2 = target2.long()#确保标签 长整型

                    loss3 = loss_function2(feature1, target)
                    loss4 = loss_function2(feature2, target2)
                    loss = loss1 + loss2 * alpha + loss3 * 0.5 +loss4 * 0.2
                    print(f"Loss calculated: {loss.item()}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    train_loss.append(loss.item())
                    print(f"Batch {i} finished")
                sum_loss.append(sum(train_loss)/len(x_train))  
                print(sum_loss[epoch])

                # evaluation on dev set 
                model.eval()
                with torch.no_grad():
                    pred1,_,_,_ = model(x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len, x_val_input_ids2)
                    acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_val)
                    val_f1_average.append(f1_average)
                
                # evaluation on test set
                x_test_len_list = dh.sep_test_set(x_test_len)
                y_test_list = dh.sep_test_set(y_test)
                x_test_input_ids_list = dh.sep_test_set(x_test_input_ids)
                x_test_seg_ids_list = dh.sep_test_set(x_test_seg_ids)
                x_test_atten_masks_list = dh.sep_test_set(x_test_atten_masks)
                x_test_input_ids_list2 = dh.sep_test_set(x_test_input_ids2)
                with torch.no_grad():
                    for ind in range(len(y_test_list)):
                        pred1,_,_,_ = model(x_test_input_ids_list[ind], x_test_seg_ids_list[ind], \
                                      x_test_atten_masks_list[ind], x_test_len_list[ind], x_test_input_ids_list2[ind])
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
                        print(f"Index value: {ind}, List length: {len(test_f1_average)}")
                        # 确保列表足够长
                        while len(test_f1_average) <= ind:
                            test_f1_average.append([])
                            
                        test_f1_average[ind].append(f1_average)
                        

            best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1] 
            best_result.append([f1[best_epoch] for f1 in test_f1_average])

            print("******************************************")
            print("dev results with seed {} on all epochs".format(seed))
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])
            print("******************************************")
            print("test results with seed {} on all epochs".format(seed))
            print(test_f1_average)
            print("******************************************")
        
        # model that performs best on the dev set is evaluated on the test set
        print("model performance on the test set: ")
        print(best_result)

class Stance_loss(nn.Module):
    def __init__(self, temperature, contrast_mode='all',
                 base_temperature=0.07):
        super(Stance_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 添加调试信息
        print(f"Features shape: {features.shape}")
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
            print(f"Labels values: {labels}")
        
        if len(features.shape) < 3:
            print(f"Warning: Features shape {features.shape} has less than 3 dimensions")
            # 添加维度
            features = features.unsqueeze(1)
            print(f"Reshaped features: {features.shape}")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        print(f"Batch size: {batch_size}")
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            print(f"Reshaped labels: {labels.shape}")
            if labels.shape[0] != batch_size:
                print(f"Error: Labels count {labels.shape[0]} != batch size {batch_size}")
                # raise ValueError('Num of labels does not match num of features')
                # 调整标签数量以匹配批次大小
                min_size = min(labels.shape[0], batch_size)
                labels = labels[:min_size]
                features = features[:min_size]
                batch_size = min_size
                print(f"Adjusted batch size: {batch_size}")
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss
if __name__ == "__main__":
    run_classifier()

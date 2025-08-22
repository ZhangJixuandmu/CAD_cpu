import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer

# Tokenization
def convert_data_to_ids(tokenizer, target, target2, text):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for tar, sent in zip(target, text):
        encoded_dict = tokenizer.encode_plus(
                            tar,                            # Target to encode
                            sent,                           # Sentence to encode
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = 128,               # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))
    
    for tar, sent in zip(target2, text):
        encoded_dict = tokenizer.encode_plus(
                            tar,
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
    return input_ids, seg_ids, attention_masks, sent_len


# BERTweet tokenizer
def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select):
    
    print('Loading data')
    
    x_train,y_train,x_train_target,y_train2,x_train_target2 = x_train_all[0],x_train_all[1],x_train_all[2],\
                                                              x_train_all[3],x_train_all[4]
    x_val,y_val,x_val_target,y_val2,x_val_target2 = x_val_all[0],x_val_all[1],x_val_all[2],x_val_all[3],x_val_all[4]
    x_test,y_test,x_test_target,y_test2,x_test_target2 = x_test_all[0],x_test_all[1],x_test_all[2],\
                                                         x_test_all[3],x_test_all[4]

    print("Length of original x_train: %d, the sum is: %d"%(len(x_train), sum(y_train)))
    print("Length of original x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
    print("Length of original x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
    # get the tokenizer

    tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    # tokenization
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train_target2, x_train)
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val_target2, x_val)
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test_target2, x_test)
    
    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len,y_train2]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len,y_val2]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len,y_test2]
    
    return x_train_all,x_val_all,x_test_all


def data_load(x_all, batch_size, train_mode):

    # y2 = [1 if y[half_y+i]==y[i] else 0 for i in range(len(y[:half_y]))] * 2
    if train_mode == "unified":
        half_y = int(len(x_all[3]) / 2)
        y = x_all[3]
        y2 = [1 if y[half_y + i] == y[i] else 0 for i in range(len(y[:half_y]))] * 2
    elif train_mode == "adhoc":
        y2 = [1 if x_all[3][i] == x_all[5][i] else 0 for i in range(len(x_all[3]))]

    half = int(len(x_all[0])/2)
    # x_input_ids = torch.tensor(x_all[0][:half], dtype=torch.long).to(device)
    # x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).to(device)
    # x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).to(device)
    # y = torch.tensor(x_all[3], dtype=torch.long).to(device)
    # x_len = torch.tensor(x_all[4], dtype=torch.long).to(device)
    #检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_input_ids = torch.tensor(x_all[0][:half], dtype=torch.long).to(device)
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).to(device)
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).to(device)
    y = torch.tensor(x_all[3], dtype=torch.long).to(device)
    x_len = torch.tensor(x_all[4], dtype=torch.long).to(device)
    x_input_ids2 = torch.tensor(x_all[0][half:], dtype=torch.long).to(device)
    y2 = torch.tensor(y2, dtype=torch.long).to(device)

    # 在创建 TensorDataset 之前添加调试信息
    print(f"Shapes of tensors:")
    print(f"x_input_ids: {x_input_ids.shape if hasattr(x_input_ids, 'shape') else len(x_input_ids)}")
    print(f"x_seg_ids: {x_seg_ids.shape if hasattr(x_seg_ids, 'shape') else len(x_seg_ids)}")
    print(f"x_atten_masks: {x_atten_masks.shape if hasattr(x_atten_masks, 'shape') else len(x_atten_masks)}")
    print(f"y: {y.shape if hasattr(y, 'shape') else len(y)}")
    print(f"x_len: {x_len.shape if hasattr(x_len, 'shape') else len(x_len)}")
    print(f"x_input_ids2: {x_input_ids2.shape if hasattr(x_input_ids2, 'shape') else len(x_input_ids2)}")
    print(f"y2: {y2.shape if hasattr(y2, 'shape') else len(y2)}")
    
    # 检查所有张量的第一个维度是否相同
    shapes = [
        len(x_input_ids),
        len(x_seg_ids),
        len(x_atten_masks),
        len(y),
        len(x_len),
        len(x_input_ids2),
        len(y2)
    ]
    
    if len(set(shapes)) > 1:
        print(f"Size mismatch detected! Sizes: {shapes}")
        # 找到最小尺寸并截断所有张量
        min_size = min(shapes)
        print(f"Truncating all tensors to size: {min_size}")
        
        x_input_ids = x_input_ids[:min_size]
        x_seg_ids = x_seg_ids[:min_size]
        x_atten_masks = x_atten_masks[:min_size]
        y = y[:min_size]
        x_len = x_len[:min_size]
        x_input_ids2 = x_input_ids2[:min_size]
        y2 = y2[:min_size]

    # 检查数据是否为空
    if len(x_input_ids) == 0:
        raise ValueError("Input data is empty after processing.")
    
    # 检查序列长度
    if x_len.min() == 0:
        print("Warning: Some sequences have zero length!")
        x_len = torch.clamp(x_len, min=1)  # 确保长度不为0

    # 检查输入和目标形状是否匹配
    print(f"Input IDs shape: {x_input_ids.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Input IDs2 shape: {x_input_ids2.shape}")
    print(f"Target2 shape: {y2.shape}")
    
    # 确保形状匹配
    assert x_input_ids.shape[0] == y.shape[0], "Input and target batch size mismatch"
    assert x_input_ids2.shape[0] == y2.shape[0], "Input2 and target2 batch size mismatch"

    # 创建 TensorDataset
    tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,x_input_ids2,y2)
    data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)

    return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, x_input_ids2, y2
    

def sep_test_set(input_data):
    
    # split the combined test set for each target
    # trump_hillary数据集
    # trump_ted数据集
    # split the combined test set for each target
    data_list = [input_data[:355], input_data[890:1245], input_data[355:618],\
                 input_data[1245:1508], input_data[618:890], input_data[1508:1780]]
    return data_list
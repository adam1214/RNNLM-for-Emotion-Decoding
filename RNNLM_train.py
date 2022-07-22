# ref:https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import pdb
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

class RNNLM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        #self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, h):
        # Forward propagate LSTM
        out, h = self.rnn(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        #out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.relu(self.linear_1(out))
        #out = self.relu(self.linear_2(out))
        out = self.dropout(out)
        out = self.linear_3(out)
        return out, h

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def train_val(input_tensor, label_tensor, val=0):
    global best_val_uar, best_epoch
    
    losses_list = []
    preds, labels = [], []
    for batch_index in range(0, input_tensor.shape[0], batch_size):
        if (batch_index + batch_size) >= input_tensor.shape[0]:
            inputs = input_tensor[batch_index:, :, :].to(device)
            targets = label_tensor[batch_index:, :].to(device, dtype=torch.long)
            states = torch.zeros(num_layers, input_tensor.shape[0]-batch_index, hidden_size).to(device)
        else:
            inputs = input_tensor[batch_index:batch_index+batch_size, :, :].to(device)
            targets = label_tensor[batch_index:batch_index+batch_size:, :].to(device, dtype=torch.long)
            states = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            
        outputs, states = model(inputs, states)
        
        loss = loss_function(outputs.permute(0,2,1), targets)
        losses_list.append(loss.item())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        targets = targets.cpu().numpy().tolist()
        outputs = torch.argmax(outputs, dim = 2).cpu().numpy().tolist()
        preds += outputs
        labels += targets

    avg_loss = round(np.sum(losses_list) / len(losses_list), 4)
    new_preds, new_labels = [], []
    for i,label in enumerate(labels):
        for j,l in enumerate(label):
            if l != -1:
                new_labels.append(l)
                new_preds.append(preds[i][j])
        
    avg_uar = recall_score(new_labels, new_preds, average='macro') * 100
    
    if val == 1 and avg_uar > best_val_uar:
        best_val_uar = avg_uar
        best_epoch = epoch
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
        torch.save(checkpoint, './model/' + args.dataset + '/best_model_fold' + str(args.model_num) + '.pth')
    
    return avg_loss, avg_uar

if __name__ == "__main__":
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.01
    num_layers = 1
    hidden_size = 60
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', "--dataset", type=str, help="which dataset?", default='NNIME')
    #parser.add_argument('-m', "--modality", type=str, help="which pre-trained modality?", default='audio')
    parser.add_argument("-n", "--model_num", type=int, help="which model number you want to train?", default=5)
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == 'IEMOCAP':
        emo_all = joblib.load('./data/IEMOCAP/emo_all_iemocap.pkl')
        dialogs_edit = joblib.load('./data/IEMOCAP/dialog_rearrange_4emo_iemocap.pkl')
        #pretrained_output = joblib.load('./data/IEMOCAP/DAG_outputs_4_all_fold_' + args.modality + '.pkl')
    else:
        emo_all = joblib.load('./data/NNIME/emo_all.pkl')
        dialogs_edit = joblib.load('./data/NNIME/dialogs_4emo.pkl')
        #pretrained_output = joblib.load('./data/NNIME/DAG_outputs_4_all_fold_interspeech_' + args.modality + '.pkl')
        
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    emo_to_ix = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3, 'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, START_TAG: 4, STOP_TAG: 5}
    emo2num = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3}
    model_num_val_map = {1:'5', 2:'4', 3:'2', 4:'1', 5: '3'}
    '''
    for utt in pretrained_output:
        pretrained_output[utt] = softmax(pretrained_output[utt])
    ''' 
    pretrained_output = {}
    for utt in emo_all:
        if emo_all[utt] in ['ang', 'hap', 'neu', 'sad', 'Anger', 'Happiness', 'Neutral', 'Sadness']:
            prob_distribution = np.zeros(4)
            prob_distribution[emo2num[emo_all[utt]]] = 1.0
            pretrained_output[utt] = prob_distribution
    
    # ensure pretrained model performance
    labels = []
    predicts = []
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            labels.append(emo2num[emo_all[utt]])
            predicts.append(pretrained_output[utt].argmax())
            
    print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')
    
    dia_name_list = []
    for dialog_name in dialogs_edit:
        dia_name_list.append(dialog_name)
    
    model = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    if args.dataset == 'NNIME':
        ang_cnt, hap_cnt, neu_cnt, sad_cnt = 0, 0, 0, 0
        for utt in emo_all:
            if utt[4] != args.model_num:
                if emo_all[utt] == 'Anger':
                    ang_cnt += 1
                elif emo_all[utt] == 'Happiness':
                    hap_cnt += 1
                elif emo_all[utt] == 'Neutral':
                    neu_cnt += 1
                elif emo_all[utt] == 'Sadness':
                    sad_cnt += 1
        max_cnt = max([ang_cnt, hap_cnt, neu_cnt, sad_cnt])
        loss_function = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.FloatTensor([max_cnt/ang_cnt, max_cnt/hap_cnt, max_cnt/neu_cnt, max_cnt/sad_cnt]).to(device))
    else:
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_epoch, best_val_uar = 0, 0
    for epoch in range(num_epochs):
        # shuffle the conversation order
        random.shuffle(dia_name_list)
        train_input, train_label, val_input, val_label = [], [], [], []
        utts_order_list = []
        for dialog_name in dia_name_list:
            for i, utt in enumerate(dialogs_edit[dialog_name]):
                if dialog_name[4] != model_num_val_map[args.model_num] and dialog_name[4] != str(args.model_num): # train set
                    if i == 0:
                        train_input.append([])
                        train_label.append([])
                        
                    if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                        train_input[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                        train_label[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])

                elif dialog_name[4] == model_num_val_map[args.model_num]: # val set
                    if i == 0:
                        val_input.append([])
                        val_label.append([])
                        
                    if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                        val_input[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                        val_label[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                        utts_order_list.append(dialogs_edit[dialog_name][i+1])
        
        train_label_tensor = pad_sequence([torch.FloatTensor(seq) for seq in train_label], batch_first = True, padding_value = -1)
        train_input_tensor = pad_sequence([torch.stack(seq) for seq in train_input], batch_first = True)
        val_label_tensor = pad_sequence([torch.FloatTensor(seq) for seq in val_label], batch_first = True, padding_value = -1)
        val_input_tensor = pad_sequence([torch.stack(seq) for seq in val_input], batch_first = True)
        
        # training
        avg_loss_train, train_uar = train_val(train_input_tensor, train_label_tensor)
        # validation
        avg_loss_val, val_uar = train_val(val_input_tensor, val_label_tensor, val=1)
        print('EPOCH', epoch, 'Train loss', avg_loss_train, 'Train UAR', round(train_uar, 2), 'Val loss', avg_loss_val, 'Val UAR', round(val_uar, 2))
        print('The best epoch so far: ', best_epoch, 'The best Val UAR so far:', round(best_val_uar,2))
    print('Train finish! The best epoch: ', best_epoch)
    
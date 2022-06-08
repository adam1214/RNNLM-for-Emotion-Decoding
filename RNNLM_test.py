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
from RNNLM_train import RNNLM
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def test(input_tensor, label_tensor, model):
    global new_preds, new_labels
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
        
        targets = targets.cpu().numpy().tolist()
        #outputs = torch.argmax(outputs, dim = 2).cpu().numpy().tolist()
        outputs = outputs.detach().cpu().numpy().tolist()
        preds += outputs
        labels += targets

    for i,label in enumerate(labels):
        for j,l in enumerate(label):
            if l != -1:
                new_preds.append(preds[i][j])
                new_labels.append(labels[i][j])

if __name__ == "__main__":
    pretrained_weight = 0.7
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
    parser.add_argument('-m', "--modality", type=str, help="which pre-trained modality?", default='text_audio')
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == 'IEMOCAP':
        emo_all = joblib.load('./data/IEMOCAP/emo_all_iemocap.pkl')
        dialogs_edit = joblib.load('./data/IEMOCAP/dialog_rearrange_4emo_iemocap.pkl')
        pretrained_output = joblib.load('./data/IEMOCAP/DAG_outputs_4_all_fold_' + args.modality + '.pkl')
    else:
        emo_all = joblib.load('./data/NNIME/emo_all.pkl')
        dialogs_edit = joblib.load('./data/NNIME/dialogs_4emo.pkl')
        pretrained_output = joblib.load('./data/NNIME/DAG_outputs_4_all_fold_interspeech_' + args.modality + '.pkl')
        
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    emo_to_ix = {'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3, 'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, START_TAG: 4, STOP_TAG: 5}
    emo2num = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3, 'Anger':0, 'Happiness':1, 'Neutral':2, 'Sadness':3}
    model_num_val_map = {1:'5', 2:'4', 3:'2', 4:'1', 5: '3'}
    
    for utt in pretrained_output:
        pretrained_output[utt] = softmax(pretrained_output[utt])
    
    # ensure pretrained model performance
    labels = []
    predicts = []
    
    for dialog_name in dialogs_edit:
        for utt in dialogs_edit[dialog_name]:
            labels.append(emo2num[emo_all[utt]])
            predicts.append(pretrained_output[utt].argmax())
            
    print('pretrained UAR:', round(recall_score(labels, predicts, average='macro')*100, 2), '%')
    print('pretrained ACC:', round(accuracy_score(labels, predicts)*100, 2), '%')
    
    test_input_ses01, test_label_ses01, test_input_ses02, test_label_ses02, test_input_ses03, test_label_ses03 = [], [], [], [], [], []
    test_input_ses04, test_label_ses04, test_input_ses05, test_label_ses05 = [], [], [], []
    utts_order_list = []
    for dialog_name in dialogs_edit:
        for i, utt in enumerate(dialogs_edit[dialog_name]):
            if dialog_name[4] == '1':
                if i == 0:
                    test_input_ses01.append([])
                    test_label_ses01.append([])
                    #utts_order_list.append(dialogs_edit[dialog_name][i])
                    
                if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                    test_input_ses01[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                    test_label_ses01[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                    utts_order_list.append(dialogs_edit[dialog_name][i+1])
                    
            elif dialog_name[4] == '2':
                if i == 0:
                    test_input_ses02.append([])
                    test_label_ses02.append([])
                    #utts_order_list.append(dialogs_edit[dialog_name][i])
                    
                if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                    test_input_ses02[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                    test_label_ses02[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                    utts_order_list.append(dialogs_edit[dialog_name][i+1])
                    
            elif dialog_name[4] == '3':
                if i == 0:
                    test_input_ses03.append([])
                    test_label_ses03.append([])
                    #utts_order_list.append(dialogs_edit[dialog_name][i])
                    
                if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                    test_input_ses03[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                    test_label_ses03[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                    utts_order_list.append(dialogs_edit[dialog_name][i+1])
                    
            elif dialog_name[4] == '4':
                if i == 0:
                    test_input_ses04.append([])
                    test_label_ses04.append([])
                    #utts_order_list.append(dialogs_edit[dialog_name][i])
                    
                if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                    test_input_ses04[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                    test_label_ses04[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                    utts_order_list.append(dialogs_edit[dialog_name][i+1])
                    
            elif dialog_name[4] == '5':
                if i == 0:
                    test_input_ses05.append([])
                    test_label_ses05.append([])
                    #utts_order_list.append(dialogs_edit[dialog_name][i])
                    
                if i >= 0 and i <= (len(dialogs_edit[dialog_name])-2):
                    test_input_ses05[-1].append(torch.FloatTensor(pretrained_output[utt].tolist()))
                    test_label_ses05[-1].append(emo2num[emo_all[dialogs_edit[dialog_name][i+1]]])
                    utts_order_list.append(dialogs_edit[dialog_name][i+1])

    
    test_label_ses01_tensor = pad_sequence([torch.FloatTensor(seq) for seq in test_label_ses01], batch_first = True, padding_value = -1)
    test_input_ses01_tensor = pad_sequence([torch.stack(seq) for seq in test_input_ses01], batch_first = True)
    
    test_label_ses02_tensor = pad_sequence([torch.FloatTensor(seq) for seq in test_label_ses02], batch_first = True, padding_value = -1)
    test_input_ses02_tensor = pad_sequence([torch.stack(seq) for seq in test_input_ses02], batch_first = True)
    
    test_label_ses03_tensor = pad_sequence([torch.FloatTensor(seq) for seq in test_label_ses03], batch_first = True, padding_value = -1)
    test_input_ses03_tensor = pad_sequence([torch.stack(seq) for seq in test_input_ses03], batch_first = True)
    
    test_label_ses04_tensor = pad_sequence([torch.FloatTensor(seq) for seq in test_label_ses04], batch_first = True, padding_value = -1)
    test_input_ses04_tensor = pad_sequence([torch.stack(seq) for seq in test_input_ses04], batch_first = True)
    
    test_label_ses05_tensor = pad_sequence([torch.FloatTensor(seq) for seq in test_label_ses05], batch_first = True, padding_value = -1)
    test_input_ses05_tensor = pad_sequence([torch.stack(seq) for seq in test_input_ses05], batch_first = True)

    model_1 = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    checkpoint = torch.load('./model/' + args.dataset + '/best_model_fold1.pth')
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_1.eval()
    
    model_2 = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    checkpoint = torch.load('./model/' + args.dataset + '/best_model_fold2.pth')
    model_2.load_state_dict(checkpoint['model_state_dict'])
    model_2.eval()
    
    model_3 = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    checkpoint = torch.load('./model/' + args.dataset + '/best_model_fold3.pth')
    model_3.load_state_dict(checkpoint['model_state_dict'])
    model_3.eval()
    
    model_4 = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    checkpoint = torch.load('./model/' + args.dataset + '/best_model_fold4.pth')
    model_4.load_state_dict(checkpoint['model_state_dict'])
    model_4.eval()
    
    model_5 = RNNLM(embed_size=4, hidden_size=hidden_size, num_layers=num_layers).to(device)
    checkpoint = torch.load('./model/' + args.dataset + '/best_model_fold5.pth')
    model_5.load_state_dict(checkpoint['model_state_dict'])
    model_5.eval()
    
    # inference
    new_preds, new_labels = [], []
    test(test_input_ses01_tensor, test_label_ses01_tensor, model_1)
    test(test_input_ses02_tensor, test_label_ses02_tensor, model_2)
    test(test_input_ses03_tensor, test_label_ses03_tensor, model_3)
    test(test_input_ses04_tensor, test_label_ses04_tensor, model_4)
    test(test_input_ses05_tensor, test_label_ses05_tensor, model_5)
    '''
    print('RNNLM UAR:', round(recall_score(new_labels, new_preds, average='macro') * 100, 2))
    print('RNNLM ACC:', round(accuracy_score(new_labels, new_preds) * 100, 2))
    '''
    
    pred_prob_dict, pred_dict = {}, {}
    for i in range(len(new_preds)):
        pred_prob_dict[utts_order_list[i]] = softmax(np.array(new_preds[i], dtype=np.float64))
    
    for dia in dialogs_edit:
        for utt in dialogs_edit[dia]:
            pred_prob_dict[utt] = pretrained_output[utt]
            break
    for utt in pred_prob_dict:
        pred_prob_dict[utt] = (1-pretrained_weight) * pred_prob_dict[utt] + pretrained_weight * pretrained_output[utt]
        pred_dict[utt] = pred_prob_dict[utt].argmax()
    
    joblib.dump(pred_dict, './model/' + args.dataset + '/preds_4.pkl')
    joblib.dump(pred_prob_dict, './model/' + args.dataset + '/RNNLM_pred_prob_dict_' + args.modality + '.pkl')
    
    preds, labels = [], []
    for utt in pred_dict:
        preds.append(pred_dict[utt])
        labels.append(emo2num[emo_all[utt]])
    print(len(labels), len(preds))
    print('RNNLM UAR:', round(recall_score(labels, preds, average='macro') * 100, 2), '%')
    print('RNNLM ACC:', round(accuracy_score(labels, preds) * 100, 2), '%')
    

    
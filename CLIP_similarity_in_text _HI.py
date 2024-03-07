import torch
import torch.nn as nn
import pandas as pd
from Preprocessing_text import preprocessing_text
from PIL import Image
from transformers import CLIPProcessor, CLIPModel,CLIPTokenizer,AdamW
from CLIP_dataset import CLIP_Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
from CLIP_model import Classifier
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import f1_score
from hierachical_f1 import dag

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def multi_label_metrics(predictions,labels,threshold=0.5):
    #probs=F.sigmoid(torch.tensor(predictions))
    probs=F.softmax(torch.tensor(predictions))
    y_pred=np.zeros(probs.shape)
    y_pred[np.where(probs>=threshold)]=1
    
    
    y_true=labels
    f1=f1_score(y_true,y_pred,average='micro')
    
    return f1

def h_f1(outputs, labels,threshold,id2label):
    
    h_f=0
    for output,label in zip(outputs,labels):
        pred_an=set()
        true_an=set()
        #pred=np.where(F.sigmoid(output).detach().cpu().numpy()>=threshold)
        pred=np.where(F.softmax(output).detach().cpu().numpy()>=threshold)
        true_label=np.where(label.detach().cpu().numpy()==1)
        #print(pred,true_label)
        for node in pred[0]:
            pred_an.add(id2label[node])
            pred_an=pred_an | dag.get_ancestors(id2label[node])
        for node in true_label[0]:
            true_an.add(id2label[node])
            true_an=true_an | dag.get_ancestors(id2label[node])
            
        intersection=pred_an & true_an
        if len(pred_an)!=0 and len(true_an)!=0:
            h_p=len(intersection)/len(pred_an)
            h_r=len(intersection)/len(true_an)
            h_f+=2*h_p*h_r/(h_p+h_r)
        else:
            h_f+=0
      
    h_f/=len(outputs)
    return h_f

def train_fn(train_dataloaer,model,optimizer,criterion,device,id2label):
    
    model.train()
    total_loss=0.0
    total_f1=0.0
    threshold=0.05
    for i,batch in enumerate(train_dataloader):
        
        
        label=batch['labels'].to(device)
        output=model(batch)
        loss=criterion(output,label)
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i%100==0:
            print(i, total_loss)
        total_f1+=h_f1(output,label,threshold,id2label)
        
        
        #preds.extend(output.detach().cpu().numpy())  
        #true_label.extend(label.detach().cpu().numpy())  
    
    f1=total_f1/len(train_dataloader)
    avg_train_loss=total_loss/len(train_dataloader)
    
    return total_loss,avg_train_loss,f1

def valid_fn(valid_dataloader,model,criterion,device,id2label):
    
    model.eval()
    total_loss=0.0
    total_f1=0.0
    threshold=0.05
    with torch.no_grad():
        for batch in valid_dataloader:
            label=batch['labels'].to(device)
            output=model(batch)
            
            loss=criterion(output,label)
            total_loss+=loss.item()
            
            total_f1+=h_f1(output,label,threshold,id2label) 
   
    
    f1=total_f1/len(valid_dataloader)
    avg_valid_loss=total_loss/len(valid_dataloader)
    return total_loss,avg_valid_loss,f1

def experiment_fn(train_dataloader,valid_dataloader,device,model_name,n_labels,id2label):
    
    encoder = CLIPModel.from_pretrained(model_name)
    model=Classifier(encoder,n_labels,device).to(device)
    optimizer=AdamW(model.parameters(),lr=1e-5,eps=1e-8)
    #criterion=nn.BCEWithLogitsLoss().to(device)
    criterion=nn.CrossEntropyLoss().to(device)
    epoch=20
    for ep in range(epoch):
        
        train_loss,avg_train_loss,train_acc=train_fn(train_dataloader,model,optimizer,criterion,device,id2label)
        valid_loss,avg_valid_loss,valid_acc=valid_fn(valid_dataloader,model,criterion,device,id2label)
        
        print(f"EP: {ep} , train_acc : {train_acc}, valid_acc : {valid_acc}")

if __name__=="__main__":    
    
    seed=42
    set_seed(seed)    
    n_labels=23
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path="/home/labuser/Semeval/SemEval_task4/Data/subtask2a_annotation/"
    train_data=pd.read_json(data_path+"train.json",lines=False)
    #train_data.drop(train_data[train_data['id']==79200].index,inplace=True)
    
    valid_data=pd.read_json(data_path+"validation.json",lines=False)
    
    train_text=train_data['text']
    valid_text=valid_data['text']
    
    train_image=train_data['image']
    valid_image=valid_data['image']
    
    train_label=train_data['labels']
    valid_label=valid_data['labels']
  
    label2id={"Presenting Irrelevant Data (Red Herring)":0,"Misrepresentation of Someone's Position (Straw Man)":1,"Whataboutism":2,"Causal Oversimplification":3,
                      "Obfuscation, Intentional vagueness, Confusion":4,"Appeal to authority":5,"Black-and-white Fallacy/Dictatorship":6,"Name calling/Labeling":7,
                      "Loaded Language":8,"Exaggeration/Minimisation":9,"Flag-waving":10,"Doubt":11,"Appeal to fear/prejudice":12,"Slogans":13,"Thought-terminating cliché":14,
                      "Bandwagon":15,"Reductio ad hitlerum":16,"Repetition":17,"Smears":18,"Glittering generalities (Virtue)":19,"Transfer":20,"Appeal to (Strong) Emotions":21}
    
    id2label={}
    for key,value in label2id.items():
        id2label[value]=key
        
    
    """
    for l in label2id.keys():
        labels_num = train_data['labels'].str.contains(str(l),regex=False).sum()
        print(f'train : the number of label {l}: {labels_num}, the ratio : {labels_num/len(train_data)*100:.1f}% ')
        
    
    
    for l in label2id.keys():
        v_labels_num = valid_data['labels'].str.contains(str(l),regex=False).sum()
        
        print(f'valid : the number of label {l}: {v_labels_num}, the ratio : {v_labels_num/len(valid_data)*100:.1f}% ')
        
 """   
    train_image_path="/home/labuser/Semeval/SemEval_task4/Data/train_images/"
    valid_image_path="/home/labuser/Semeval/SemEval_task4/Data/validation_images/"
    train_pre_text=preprocessing_text(train_text)
    valid_pre_text=preprocessing_text(valid_text)
    
    model_name="openai/clip-vit-base-patch32"
    processor=CLIPProcessor.from_pretrained(model_name)
  
    train_dataset=CLIP_Dataset(train_pre_text,train_image,train_label,label2id,processor,train_image_path)
    valid_dataset=CLIP_Dataset(valid_pre_text,valid_image,valid_label,label2id,processor,valid_image_path)
    
    
    batch_size=8
    
    
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)
    

    experiment_fn(train_dataloader,valid_dataloader,device,model_name,n_labels,id2label)
   
        

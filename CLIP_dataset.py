import torch.nn as nn
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader
from transformers import CLIPProcessor
from PIL import Image



class CLIP_Dataset(Dataset):
    
    def __init__(self,texts,images,labels,label2id,processor,image_path):
        
        super().__init__()
        self.texts=texts
        self.images=images
        self.labels=labels
        self.classes=label2id
        self.processor=processor
       
        self.image_path=image_path

    def __len__(self):
    
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        txt=self.texts[idx]
        #print(self.images[idx])
        img=Image.open(self.image_path+self.images[idx])
        
        text_inputs = self.processor(text=txt, images=None, return_tensors="pt", max_length=77,truncation=True,padding='max_length')
        img_inputs=self.processor(text=None,images=img,padding=True,return_tensors="pt")
        one_hot=[0]*23
        for label in self.labels[idx]:
            one_hot[self.classes[label]]=1
        labels=[self.classes[l] for l in self.labels[idx]]
        
        return {'input_ids':text_inputs['input_ids'].flatten(),
                'attention_mask':text_inputs['attention_mask'].flatten(),
                'pixel_values':img_inputs['pixel_values'],
                'labels':torch.tensor(one_hot,dtype=torch.float32)}
        
        
        
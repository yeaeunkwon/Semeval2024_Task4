import torch.nn as nn
import torch


class Classifier(nn.Module):
    
    def __init__(self,encoder,n_labels,device):
        super().__init__()
        self.clip=encoder
        self.fc=nn.Linear(1024,n_labels)
        self.device=device
    
        
    def forward(self,data):
        output=self.clip(input_ids=data['input_ids'].to(self.device),attention_mask=data['attention_mask'].to(self.device),pixel_values=torch.squeeze(data['pixel_values'],1).to(self.device))
        text_emb=output.text_embeds
        img_emb=output.image_embeds
        
        concat=torch.concat((text_emb,img_emb),dim=1)
        output=self.fc(concat)
        
        
        
        return output
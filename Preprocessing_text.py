import torch
import pandas as pd
import re

def preprocessing_text(texts):
    # remove the words other than english and number
    # lower case 
    processed_text=[]
    for text in texts:
        clean_text=text.replace("\\n", " ")
        clean_text=clean_text.lower().strip()
        clean_text=re.sub("[^0-9a-z ]","",clean_text)
        processed_text.append(clean_text)
    return processed_text

        
    
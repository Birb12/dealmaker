import pandas as pd
import re
import csv
from langdetect import detect
import torch
from torch.utils.data import Dataset


cleandescription = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') # regex to clean html, including tags enclosed in <> and tags like &nsbm

def cleanhtml(html):
    return re.sub(cleandescription, ' ', html)

class DealsDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length, path="datafortrain.csv"): #FIXME change path for your data
        try:
            self.path = path
            self.df = pd.read_csv(path, index_col=False)
        except:
            print("!Invalid file path!")

        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for txt in txt_list:
            encodings_dict = tokenizer(
                '<|startoftext|' + txt + '<|endoftext|>',
                truncation=True,
                max_length=max_length,
                padding="max_length")
            input_ids = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)
    
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.attn_masks[idx]
    
    def __len__(self):
        return len(self.input_ids)

    def clean_dataset(self): # get rid of html and spanish rows from original dataset, get rid of unneeded columns, get rid of empty values, rename columns

        self.df.drop( 
            ['newParentVersionId',
            'merchantOffers',
            'shortcodeType',
            'metadata.parentId'], axis=1, inplace=True
        )

        self.df.dropna(axis=0, inplace=True)

        self.df.rename({'description' : 'OUTPUT', 'hed': 'PRODUCT', 'subhed': 'DISCOUNT'}, inplace=True, axis=1)
        print(self.df.columns)
        for i in self.df.itertuples():
            if detect(self.df.at[i.Index, "OUTPUT"]) == 'es' or "%" not in self.df.at[i.Index, "DISCOUNT"] and "$" not in self.df.at[i.Index, "DISCOUNT"]: # drop spanish or rows with no discount
                self.df.drop(i.Index, axis=0, inplace=True)
            else:
                self.df.at[i.Index, "OUTPUT"] = cleanhtml(self.df.loc[i.Index, "OUTPUT"]) # clean html
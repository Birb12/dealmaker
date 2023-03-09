import pandas as pd
import re
import csv
from langdetect import detect
import torch
from torch.utils.data import Dataset


cleandescription = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});') # regex to clean html, including tags enclosed in <> and tags like &nsbm

def cleanhtml(html):
    return re.sub(cleandescription, ' ', html)

class PromptsDataset():
    def __init__(self, path="datafortrain.csv"): #FIXME change path for your data
        try:
            self.path = path
            self.df = pd.read_csv(path, index_col=False)
        except:
            print("!Invalid file path!")

        self.prompts = pd.DataFrame({'prompt': pd.Series(dtype='str'), 'completion':pd.Series(dtype='str')})
    def generate_prompts(self): # generate prompts for GPT, this is then converted into a .jsonl file using the openai CLI
        for i in self.df.itertuples():
            self.prompts.at[i.Index, 'prompt'] = "Make a description about an " + self.df.at[i.Index, 'PRODUCT'] + " with an " + self.df.at[i.Index, 'DISCOUNT'] + " discount"
            self.prompts.at[i.Index, 'completion'] = self.df.at[i.Index, 'OUTPUT']

        self.prompts.to_json("promptsforgpt.jsonl", orient='records', lines=True)


    def clean_dataset(self): # get rid of html and spanish rows from original dataset, get rid of unneeded columns, get rid of empty values, rename columns

        self.df.drop( 
            ['newParentVersionId',
            'merchantOffers',
            'shortcodeType',
            'metadata.parentId'], axis=1, inplace=True
        )

        self.df.dropna(axis=0, inplace=True)

        self.df.rename({'description' : 'OUTPUT', 'hed': 'PRODUCT', 'subhed': 'DISCOUNT'}, inplace=True, axis=1)
        for i in self.df.itertuples():
            if detect(self.df.at[i.Index, "OUTPUT"]) == 'es' or "%" not in self.df.at[i.Index, "DISCOUNT"] and "$" not in self.df.at[i.Index, "DISCOUNT"]: # drop spanish or rows with no discount
                self.df.drop(i.Index, axis=0, inplace=True)
            else:
                self.df.at[i.Index, "OUTPUT"] = cleanhtml(self.df.loc[i.Index, "OUTPUT"]) # clean html


dataset = PromptsDataset("datafortrain.csv") 
dataset.generate_prompts()

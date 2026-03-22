"""Used dataset with format
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   id              57477 non-null  int64 
 1   model_a         57477 non-null  object
 2   model_b         57477 non-null  object
 3   prompt          57477 non-null  object
 4   response_a      57477 non-null  object
 5   response_b      57477 non-null  object
 6   winner_model_a  57477 non-null  int64 
 7   winner_model_b  57477 non-null  int64 
 8   winner_tie      57477 non-null  int64 
dtypes: int64(4), object(5)

prompt, response_a and response_b are "lists of string" saved as string
"""

import argparse

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from data_preprocessing import prepare
from tokenizer import robust_tokenize

class ScorerDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.input_ids_1 = []
        self.input_ids_2 = []
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Tokenizing"):
            prompt = [prompt1 if prompt1 else "" for prompt1 in row["prompt"]]
            responses_1 = [response if response else "" for response in row["response_a"]]
            responses_2 = [response if response else "" for response in row["response_b"]]

            assert(len(prompt)==len(responses_1))
            assert(len(prompt)==len(responses_2))
            enc1 = {'input_ids': [],
                    'attention_mask': [],
                    'token_type_ids': []}
            enc2 = {'input_ids': [],
                    'attention_mask': [],
                    'token_type_ids': []}
            for i in range(len(prompt)):
                enc1_1 = robust_tokenize(prompt[i], responses_1[i], tokenizer=tokenizer, max_length=512, padding=True)
                enc2_1 = robust_tokenize(prompt[i], responses_2[i], tokenizer=tokenizer, max_length=512, padding=True)
                enc1['input_ids'].append(enc1_1['input_ids'])
                enc1['attention_mask'].append(enc1_1['attention_mask'])
                enc1['token_type_ids'].append(enc1_1['token_type_ids'])

                enc2['input_ids'].append(enc2_1['input_ids'])
                enc2['attention_mask'].append(enc2_1['attention_mask'])
                enc2['token_type_ids'].append(enc2_1['token_type_ids'])

            enc1['input_ids'] = torch.stack(enc1['input_ids'])
            enc1['attention_mask'] = torch.stack(enc1['attention_mask'])
            enc1['token_type_ids'] = torch.stack(enc1['token_type_ids'])

            enc2['input_ids'] = torch.stack(enc2['input_ids'])
            enc2['attention_mask'] = torch.stack(enc2['attention_mask'])
            enc2['token_type_ids'] = torch.stack(enc2['token_type_ids'])
            
            self.input_ids_1.append(enc1)
            self.input_ids_2.append(enc2)
    
        self.labels = torch.tensor(
            data[["winner_model_a", "winner_model_b", "winner_tie"]].values,
            dtype=torch.float
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids_1[idx],
            self.input_ids_2[idx],
            self.labels[idx]
        )

def create_dataset(tokenizer, data_path="data.csv"):
    df = prepare(tokenizer, data_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = ScorerDataset(train, tokenizer)
    test_dataset = ScorerDataset(test, tokenizer)
    return train_dataset, test_dataset
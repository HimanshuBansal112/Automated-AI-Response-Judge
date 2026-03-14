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
import pandas as pd
from transformers import AutoTokenizer
import json
from tqdm import tqdm

def safe_json(cell):
    try:
        if isinstance(cell, list):
            return cell
        return json.loads(cell)
    except Exception:
        return [cell] if cell else []        

# Adds extra tokens such as question and answer
def formatting_text(row):
    output = {}
    output["response_1"] = []
    output["response_2"] = []
    for index in range(len(row["prompt"])):
        prompt = row["prompt"][index]
        response_a = row["response_a"][index]
        response_b = row["response_b"][index]
        output["response_1"].append(f"""Question: {prompt} Answer: {response_a}""")
        output["response_2"].append(f"""Question: {prompt} Answer: {response_b}""")
    return output

def filter_dataset(row, tokenizer, max_seq_len):
    length_1 = 0
    length_2 = 0
    for response_1 in row["response_1"]:
        temp_len = len(tokenizer(response_1)["input_ids"])
        if temp_len>500:
            return False
        if temp_len%max_seq_len!=0:
            temp_len = max_seq_len + (temp_len - (temp_len%max_seq_len))
        length_1 += temp_len
    for response_2 in row["response_2"]:
        temp_len = len(tokenizer(response_2)["input_ids"])
        if temp_len>500:
            return False
        if temp_len%max_seq_len!=0:
            temp_len = max_seq_len + (temp_len - (temp_len%max_seq_len))
        length_2 += temp_len
    if length_1 > 1100 or length_2 > 1100:
        return False
    return True

def prepare(tokenizer, data_path="data.csv"):
    df = pd.read_csv(data_path, quotechar='"', engine='python')

    list_cols = ["prompt", "response_a", "response_b"]

    for col in list_cols:
        df[col] = df[col].apply(safe_json)

    formatted = df.apply(formatting_text, axis=1, result_type="expand")
    df[["response_1", "response_2"]] = formatted

    max_seq_len = 512

    tqdm.pandas()

    keep_or_not = df.progress_apply(lambda row: filter_dataset(row, tokenizer, max_seq_len), axis=1)
    df = df[keep_or_not]
    df = df.drop(columns=['id', 'model_a', 'model_b', 'response_1', 'response_2'])
    return df

#Dataset will save list as string on saving
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()

    print("Dataset will save list as string on saving")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False, cache_dir="./models")
    df = prepare(tokenizer, args.data_path)
    df.to_csv("processed_data.csv", index=False)
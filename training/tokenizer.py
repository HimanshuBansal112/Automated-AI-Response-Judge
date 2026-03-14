import torch

def robust_tokenize(text, text_pair=None, tokenizer=None, max_length=None, padding=False):
    enc_a = tokenizer(text, add_special_tokens=False)['input_ids']
    
    if text_pair is not None:
        enc_b = tokenizer(text_pair, add_special_tokens=False)['input_ids']
        
        if max_length is not None:
            while len(enc_a) + len(enc_b) + 3 > max_length:
                if len(enc_a) > len(enc_b):
                    enc_a.pop()
                else:
                    enc_b.pop()
                    
        input_ids = [1] + enc_a + [2] + enc_b + [2]
        
        token_type_ids = [0] * (len(enc_a) + 2) + [1] * (len(enc_b) + 1)

    else:
        if max_length is not None:
            enc_a = enc_a[:max_length - 2]
            
        input_ids = [1] + enc_a + [2]
        token_type_ids = [0] * len(input_ids)

    attention_mask = [1] * len(input_ids)

    if padding and max_length is not None:
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
    }
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

def prepare_for_style(prompt, response_a, response_b, tokenizer):
    prompt = [prompt1 if prompt1 else "" for prompt1 in prompt]
    responses_1 = [response if response else "" for response in response_a]
    responses_2 = [response if response else "" for response in response_b]

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
    
    return enc1, enc2

def prepare_for_fact(prompt, response_a, response_b, processor):
    response_1 = ""
    response_2 = ""
    data = {
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b
    }
    for index in range(len(data["prompt"])):
        prompt = data["prompt"][index]
        response_a = data["response_a"][index]
        response_b = data["response_b"][index]
        response_1+=f"""Question: {prompt} Answer: {response_a}"""
        response_2+=f"""Question: {prompt} Answer: {response_b}"""

    instruction = """Read the following answer carefully and grade its factual correctness like a teacher. 
Score from 0 to 100, where 0 = completely incorrect, 50 = partially correct, 100 = fully correct. 
Round to the nearest integer. Output only the number, no letters or extra text."""
    dict_response_1 = [
        {"role": "system", "content": [
            {"type": "text", "text": instruction}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": response_1}
        ]}
    ]
    dict_response_2 = [
        {"role": "system", "content": [
            {"type": "text", "text": instruction}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": response_2}
        ]}
    ]

    enc1 = processor.apply_chat_template(
        dict_response_1,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    enc2 = processor.apply_chat_template(
        dict_response_2,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    return enc1, enc2
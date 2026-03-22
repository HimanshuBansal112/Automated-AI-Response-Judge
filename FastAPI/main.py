import torch
import torch.nn.functional as F
import torch_directml
from fastapi import FastAPI
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from tools import prepare_for_fact, prepare_for_style

app = FastAPI()

device = None
model_style = None
tokenizer_style = None
model_fact = None
processor_fact = None

@app.on_event("startup")
def load_model_style():
    global device, model_style, tokenizer_style, model_fact, processor_fact
    
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    torch.set_grad_enabled(False)
    #device = torch_directml.device()
    device = torch.device("cpu")
    
    tokenizer_style = AutoTokenizer.from_pretrained(
        "Himanshu167/AI-Response-Comparer",
        trust_remote_code=True,
        cache_dir="./models"
    )
    
    processor_fact = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", cache_dir="./models")

    model_style = AutoModel.from_pretrained(
        "Himanshu167/AI-Response-Comparer",
        trust_remote_code=True,
        cache_dir="./models"
    )
    
    model_fact = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-2B", cache_dir="./models")
    
    model_style = model_style.float()
    
    model_style = model_style.to(device)
    model_style.eval()
    model_fact.eval()

@app.post("/style_predict")
def style_predict(prompt: list[str], response_a: list[str], response_b: list[str]) -> list[float]:
    tokenized_response_1, tokenized_response_2 = prepare_for_style(prompt, response_a, response_b, tokenizer_style)
    
    tokenized_response_1 = {k: v.to(device) for k, v in tokenized_response_1.items()}
    tokenized_response_2 = {k: v.to(device) for k, v in tokenized_response_2.items()}

    chunk_size = torch.tensor([tokenized_response_1["input_ids"].shape[0]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model_style(tokenized_response_1, tokenized_response_2, chunk_size)
        output = F.softmax(output, dim=1)[0].detach().cpu().numpy()

    return output.tolist()

def batch_fact_predict(prompt: list[str], response_a: list[str], response_b: list[str]) -> list[float]:
    tokenized_response_1, tokenized_response_2 = prepare_for_fact(prompt, response_a, response_b, processor_fact)
    
    tokenized_response_1 = {k: v.to(device) for k, v in tokenized_response_1.items()}
    tokenized_response_2 = {k: v.to(device) for k, v in tokenized_response_2.items()}

    with torch.no_grad():
        output = model_fact.generate(**tokenized_response_1, max_new_tokens=3, pad_token_id=processor_fact.tokenizer.eos_token_id)
        A = processor_fact.decode(output[0][tokenized_response_1["input_ids"].shape[-1]:], skip_special_tokens=True)

        output = model_fact.generate(**tokenized_response_2, max_new_tokens=3, pad_token_id=processor_fact.tokenizer.eos_token_id)
        B = processor_fact.decode(output[0][tokenized_response_2["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        A = str(A).strip()
        B = str(B).strip()
        
        if str.isnumeric(B) and str.isnumeric(A):
            A = int(A)
            B = int(B)
        else:
            A = -100
            B = -100
        output = [A,B,min(A,B)]
        if any(x < 0 for x in output):
            return [-1, -1, -1]
        output_sum = sum(output)
        if output_sum == 0:
            output_sum = 1
        output = [x / output_sum for x in output]
        return output
    raise Exception("Output shouldn't have skipped")

@app.post("/fact_predict")
def iterative_fact_predict(prompt: list[str], response_a: list[str], response_b: list[str]) -> dict[str, list[float]]:
    result = [0,0,0]
    
    for i in range(len(prompt)):
        result1 = batch_fact_predict([prompt[i]], [response_a[i]], [response_b[i]])
        if result1[0] == -1:
            return [-1, -1, -1]
        result[0] += result1[0]
        result[1] += result1[1]
        result[2] += result1[2]
    
    result_sum = sum(result)
    if result_sum == 0:
        result_sum = 1
    result[0] /= result_sum
    result[1] /= result_sum
    result[2] /= result_sum
    return result

@app.post("/full_predict")
def full_predict(prompt: list[str], response_a: list[str], response_b: list[str]) -> dict[str, list[float]]:
    return {
        "Style": style_predict(prompt, response_a, response_b),
        "Fact": iterative_fact_predict(prompt, response_a, response_b)
    }
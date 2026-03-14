import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix

from custom_dataset import create_dataset

class Comparer(nn.Module):
    def __init__(self, model1, seq_len):
        super(Comparer, self).__init__()
        self.main_layer = model1
        self.linear_tokens = nn.Linear(768, 100) 
        self.linear_tokens2 = nn.Linear(200, 50)
        self.linear1 = nn.Linear(50, 3)
        self.dropout = nn.Dropout(0.1)

    def segment_mean(self, values, lengths):
        chunks = torch.split(values, lengths)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks])

    def forward(self, tokens1, tokens2, chunk_sizes):
        h1 = self.main_layer(**tokens1).last_hidden_state[:, 0, :]
        h2 = self.main_layer(**tokens2).last_hidden_state[:, 0, :]
        
        x1 = self.segment_mean(h1, chunk_sizes)
        x2 = self.segment_mean(h2, chunk_sizes)

        x1 = torch.tanh(self.linear_tokens(x1))
        x2 = torch.tanh(self.linear_tokens(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(torch.tanh(self.linear_tokens2(x)))
        
        return self.linear1(x)

def train_one_epoch(train_loader, model, device, optimizer, criterion, scaler, scheduler, accumulation_steps = 4):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.
    log_steps = 1000

    device_type = "cuda" if device.type == "cuda" else "cpu"
    is_cuda = (device_type == "cuda")

    model.train() 
    optimizer.zero_grad()
    
    for i, (input_1_og, input_2_og, labels_og) in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training")):
        input_1 = {k: v for k, v in input_1_og.items()}
        input_2 = {k: v for k, v in input_2_og.items()}

        labels = labels_og.clone().to(device)
        chunk_sizes = [len(chunk) for chunk in input_1["input_ids"]]
        input_1["input_ids"] = torch.cat(input_1["input_ids"], dim=0).to(device)
        input_2["input_ids"] = torch.cat(input_2["input_ids"], dim=0).to(device)

        input_1["attention_mask"] = torch.cat(input_1["attention_mask"], dim=0).to(device)
        input_2["attention_mask"] = torch.cat(input_2["attention_mask"], dim=0).to(device)

        smooth = 0.1
        labels_smooth = labels * (1 - smooth) + smooth / 3

        labels_smooth = labels_smooth.to(device)

        with torch.amp.autocast(dtype=torch.float16, device_type=device_type):
            outputs = model(input_1, input_2, chunk_sizes)
            loss = F.kl_div(F.log_softmax(outputs, dim=1), labels_smooth, reduction='batchmean')
            loss /= accumulation_steps

        scaler.scale(loss).backward()

        running_loss += loss.item()
        total_loss += loss.item() * accumulation_steps
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
                
            del input_1, input_2, outputs, labels
        
        if i % log_steps == log_steps - 1:
            last_loss = running_loss * accumulation_steps / log_steps
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    if len(train_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        del input_1, input_2, outputs, labels
        torch.cuda.empty_cache()
    
    return total_loss / len(train_loader.dataset)

def eval_one_epoch(test_loader, model, device):
    global output_array, label_array
    output_array = []
    label_array = []
    log_steps = 1000
    zero_tensor = torch.tensor(0.0, dtype=torch.float, device=device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    is_cuda = (device_type == "cuda")
    
    correct_count = 0

    model.eval()

    for i, (input_1_og, input_2_og, labels) in enumerate(tqdm(test_loader, total=len(test_loader), desc="Evaluating")):        
        input_1 = {k: v for k, v in input_1_og.items()}
        input_2 = {k: v for k, v in input_2_og.items()}

        labels = labels.to(device)
        chunk_sizes = [len(chunk) for chunk in input_1["input_ids"]]
        input_1["input_ids"] = torch.cat(input_1["input_ids"], dim=0).to(device)
        input_2["input_ids"] = torch.cat(input_2["input_ids"], dim=0).to(device)

        input_1["attention_mask"] = torch.cat(input_1["attention_mask"], dim=0).to(device)
        input_2["attention_mask"] = torch.cat(input_2["attention_mask"], dim=0).to(device)

        with torch.no_grad():
            outputs = model(input_1, input_2, chunk_sizes)
        outputs = outputs.argmax(dim=1)        
        
        [output_array.append(output.detach().cpu().numpy()) for output in outputs]
        
        labels = labels.argmax(dim=1)
        [label_array.append(label.detach().cpu().numpy()) for label in labels]
        
        correct_count += (outputs == labels).sum().item()
            
        if (i+1)%log_steps==0:
            print(f"Batch {i+1} Accuracy:", correct_count*100/(i*len(labels)))

    output_array = np.array(output_array)
    label_array = np.array(label_array)
    return correct_count*100/len(test_loader.dataset)

def train(num_epochs=3, data_path="data.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_model = AutoModel.from_pretrained("microsoft/deberta-v3-base", cache_dir="./models")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False, cache_dir="./models")
    
    train_dataset, test_dataset = create_dataset(tokenizer, data_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            {
                "input_ids": [x[0]["input_ids"] for x in batch],
                "attention_mask": [x[0]["attention_mask"] for x in batch],
            },
            {
                "input_ids": [x[1]["input_ids"] for x in batch],
                "attention_mask": [x[1]["attention_mask"] for x in batch],
            },
            torch.stack([x[2] for x in batch])
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            {
                "input_ids": [x[0]["input_ids"] for x in batch],
                "attention_mask": [x[0]["attention_mask"] for x in batch],
            },
            {
                "input_ids": [x[1]["input_ids"] for x in batch],
                "attention_mask": [x[1]["attention_mask"] for x in batch],
            },
            torch.stack([x[2] for x in batch])
        )
    )

    main_model = main_model.float()
    model = Comparer(main_model, 512)

    model = model.to(device)

    if main_model.dtype == torch.float16:
        optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-6)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        print("32 bit")
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)
    total_steps = len(train_loader) * num_epochs

    num_warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )

    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(train_loader, model, device, optimizer, criterion, scaler, scheduler)
        accuracy = eval_one_epoch(test_loader, model, device)
        print(f"Epoch {epoch}/{num_epochs}: {accuracy}")
        
        cm = confusion_matrix(label_array, output_array)
        print("Confusion matrix", cm)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "accuracy": accuracy
        }

        torch.save(checkpoint, "checkpoint_latest.pt")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, "checkpoint_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()
    
    train(args.num_epochs, args.data_path)
from emotion_classifier import crossvalidationdata, dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batches):
    input_ids = []
    attention_mask = []
    labels = []
    for batch in batches:
        input_ids.append(batch["input_ids"])
        attention_mask.append(batch["attention_mask"])
        labels.append(batch["labels"])
    
    return {
        "input_ids": torch.stack(input_ids).squeeze().to(device),
        "attention_mask": torch.stack(attention_mask).squeeze().to(device),
        "labels": torch.tensor(labels).to(device)
    }
          

def compute_metrics(gt, pred):
    return {
        "f1_score": f1_score(gt, pred, average="weighted"), 
        "precision_score": precision_score(gt, pred, average="weighted"),
        "recall_score": recall_score(gt, pred, average="weighted")
    }

            

data = crossvalidationdata("/Users/apple/Desktop/CaseStudy/task_1/emotion_detection/dataset/tweet_emotions.csv")

train_dataset = dataset(data.train_file_path, data.label2index)
test_dataset  = dataset(data.dev_file_path, data.label2index)
# Important note  : using only 500 sample due to computation limit
sampler = SubsetRandomSampler(range(200))

train_loader = DataLoader(train_dataset, batch_size=16, drop_last=True, sampler=sampler, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=16,  drop_last=True, sampler=sampler, collate_fn=collate_fn)


model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=len(data.label2index.keys()))
model.to(device)
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_metrics = []
    print(f"Epoch {epoch}")
    for train_batch in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(**train_batch)
        loss = output.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    if epoch % 2 == 0:
        model.eval()
        test_loss = 0
        prompt = "Train loss:  {} \n Test  loss:  {}"
        scores = []
        with torch.no_grad():
            for test_batch in test_loader:
                output = model(**test_batch)
                loss = output.loss
                test_loss += loss.item()
                logits = output.logits
                pred_proba = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(pred_proba, dim=-1)
                scores.append(compute_metrics(test_batch["labels"].detach().cpu().tolist(), prediction.detach().cpu().tolist()))


        print(pd.DataFrame(scores).to_markdown())
        print(prompt.format(train_loss/len(train_loader), test_loss/len(test_loader)))


PATH_TO_MODEL="/Users/apple/Desktop/CaseStudy/task_1/emotion_detection/models"
model.save_pretrained(PATH_TO_MODEL)
train_dataset.tokenizer.save_pretrained(PATH_TO_MODEL)
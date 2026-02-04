import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/CIPPROJECT/cipdata/processed/disaster_text_clean.csv")

# Rename if needed
df = df.rename(columns={
    df.columns[0]: "text",
    df.columns[1]: "label"
})

df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

print(df.head())
print(df["label"].value_counts())


from sklearn.model_selection import train_test_split

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)


import torch
from torch.utils.data import Dataset

class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(
    DisasterDataset(train_texts, train_labels, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    DisasterDataset(val_texts, val_labels, tokenizer),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    DisasterDataset(test_texts, test_labels, tokenizer),
    batch_size=BATCH_SIZE
)

import torch.nn as nn
from transformers import AutoModel

class DeLTran15(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.classifier(x)
    
from sklearn.metrics import classification_report

model.eval()
preds, gold = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        gold.extend(labels.cpu().numpy())

print(classification_report(gold, preds, digits=4))


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

labels_np = np.array(train_labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_np),
    y=labels_np
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,     # ↓ from 8e-5
    weight_decay=0.01
)


best_val_f1 = 0
patience = 2
wait = 0

for epoch in range(8):
    model.train()
    train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            gold.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score
    val_f1 = f1_score(gold, preds, average="weighted")

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("Classification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    digits=4
))

tokenizer.save_pretrained("deltran15_tokenizer")
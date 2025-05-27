import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('/content/jigsaw-toxic-comment-train-processed-seqlen128.csv')

# Define label columns
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Drop rows with NaN in any label column
df = df.dropna(subset=label_cols)

# ✅ Reduce dataset size for faster training (10000 samples)
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Create the labels column
df['labels'] = df[label_cols].values.tolist()

# Check
print(df.head(2))

# Define labels
labels = df['labels'].tolist()

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(list(df['comment_text']), labels, test_size=0.2, random_state=42)

# Initialize DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the texts
train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Custom dataset
class ToxicCommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label_value = self.labels[idx]
        item['labels'] = torch.tensor(label_value, dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets and dataloaders
train_dataset = ToxicCommentDataset(train_enc, train_labels)
val_dataset = ToxicCommentDataset(val_enc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize DistilBERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6, problem_type="multi_label_classification").to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(2):
    model.train()
    loop = tqdm(train_loader, leave=True, mininterval=5)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Validation loop
model.eval()
predictions, true_labels = [], []
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
    labels = batch['labels'].cpu().numpy()
    predictions.extend(preds)
    true_labels.extend(labels)

# Evaluate model
print(classification_report(true_labels, predictions, target_names=label_cols))

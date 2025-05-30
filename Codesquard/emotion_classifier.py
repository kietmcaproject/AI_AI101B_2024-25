import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('tweet_emotions.csv')
print(f"Dataset loaded with {len(df)} rows")

# Reduce dataset size for faster training
sample_size = int(len(df) * 0.1)  # 10% of the data
df = df.sample(n=sample_size, random_state=42)
print(f"Reduced dataset to {len(df)} samples for faster training")

# Display basic information
print("\nFirst few rows:")
print(df.head())

print("\nEmotion distribution:")
print(df['sentiment'].value_counts())

# Map emotions to numeric labels
unique_emotions = df['sentiment'].unique()
emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
id_to_emotion = {idx: emotion for idx, emotion in enumerate(unique_emotions)}
print(f"\nFound {len(unique_emotions)} emotions: {unique_emotions}")

# Add numeric labels
df['label'] = df['sentiment'].map(emotion_to_id)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
print(f"\nSplit dataset into {len(train_df)} training and {len(test_df)} testing samples")

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function with shorter sequence length
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=64)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenize datasets
print("\nTokenizing datasets...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Load pre-trained model
print("\nLoading BERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(unique_emotions)
)

# Define training arguments with reduced parameters for faster training
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="no",
    report_to="none",
)

# Define metrics calculation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nTraining the model (quick version)...")
trainer.train()

# Evaluate the model
print("\nEvaluating the model...")
results = trainer.evaluate()
print(f"Evaluation results: {results}")

# Save the model
model_path = "./emotion_bert_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"\nModel saved to {model_path}")

# Function to predict emotions for new text
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    return id_to_emotion[predicted_class_id]

# Test the prediction function
test_texts = [
    "I'm so happy today!",
    "This makes me really angry",
    "I feel so sad and lonely",
    "I'm worried about the upcoming exam",
    "I love spending time with my family"
]

print("\nTesting prediction on new texts:")
for text in test_texts:
    emotion = predict_emotion(text)
    print(f"Text: '{text}' → Emotion: {emotion}")

print("\nEmotion classification complete!")

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

model_path = "./emotion_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

print("Loading dataset...")
df = pd.read_csv('tweet_emotions.csv')
sample_size = int(len(df) * 0.1) 
df = df.sample(n=sample_size, random_state=42)

unique_emotions = df['sentiment'].unique()
emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
id_to_emotion = {idx: emotion for idx, emotion in enumerate(unique_emotions)}

# Add numeric labels
df['label'] = df['sentiment'].map(emotion_to_id)

# Split the dataset the same way as before
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=64)

# Prepare test dataset
test_dataset = Dataset.from_pandas(test_df)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Get predictions
model.eval()
predictions = []
true_labels = []
confidence_scores = []

for batch in range(0, len(test_tokenized), 32):
    batch_data = test_tokenized[batch:batch+32]
    inputs = tokenizer(batch_data["content"], padding=True, truncation=True, 
                      return_tensors="pt", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred = torch.argmax(outputs.logits, dim=-1).numpy()
    predictions.extend(pred)
    true_labels.extend(batch_data["label"])
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    confidence_scores.extend([probs[i, pred[i]] for i in range(len(pred))])

overall_accuracy = accuracy_score(true_labels, predictions)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Generate visualizations
# 1. Performance Metrics Table
report = classification_report(
    true_labels, 
    predictions, 
    target_names=[id_to_emotion[i] for i in range(len(unique_emotions))],
    output_dict=True
)

metrics_df = pd.DataFrame(report).transpose()
print("\nPerformance Metrics:")
print(metrics_df)
metrics_df.to_csv("emotion_metrics.csv")
print("Performance metrics saved to emotion_metrics.csv")


cm = confusion_matrix(true_labels, predictions)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_norm, 
    annot=True, 
    fmt='.2f', 
    cmap='Blues',
    xticklabels=[id_to_emotion[i] for i in range(len(unique_emotions))],
    yticklabels=[id_to_emotion[i] for i in range(len(unique_emotions))]
)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion matrix saved as confusion_matrix.png")

# 3. Per-Class Accuracy Chart
plt.figure(figsize=(14, 8))
accuracies = [report[id_to_emotion[i]]['precision'] for i in range(len(unique_emotions)) 
              if id_to_emotion[i] in report]
emotion_names = [id_to_emotion[i] for i in range(len(unique_emotions)) 
                if id_to_emotion[i] in report]
bars = plt.bar(
    emotion_names,
    accuracies,
    color='skyblue'
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.01,
        f'{height:.2f}',
        ha='center', 
        va='bottom',
        rotation=0
    )

plt.xlabel('Emotion')
plt.ylabel('Precision')
plt.title('Per-Class Precision')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('per_class_precision.png', dpi=300)
print("Per-class precision chart saved as per_class_precision.png")

# NEW: 4. Performance Matrix Image (Visual Representation of Metrics)
plt.figure(figsize=(15, 10))
metrics_to_plot = metrics_df.iloc[:-3].copy()  # Exclude the avg/total rows
metrics_to_plot = metrics_to_plot[['precision', 'recall', 'f1-score']]

# Create a heatmap of the metrics
sns.heatmap(
    metrics_to_plot, 
    annot=True, 
    fmt='.2f', 
    cmap='YlGnBu',
    linewidths=.5
)
plt.title('Performance Metrics Matrix')
plt.tight_layout()
plt.savefig('performance_matrix.png', dpi=300)
print("Performance matrix image saved as performance_matrix.png")

# NEW: 5. Accuracy Table as Image
plt.figure(figsize=(10, 6))
# Create a table-like visualization
accuracy_data = {
    'Metric': ['Overall Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
    'Score': [
        overall_accuracy,
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    ]
}
acc_df = pd.DataFrame(accuracy_data)

# Turn off the axes
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Create the table
table = plt.table(
    cellText=acc_df.values,
    colLabels=acc_df.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.4, 0.2]
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

plt.title('Accuracy Metrics', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('accuracy_table.png', dpi=300)
print("Accuracy table saved as accuracy_table.png")

# NEW: 6. Confidence Distribution by Emotion
plt.figure(figsize=(14, 8))

# Create a DataFrame with predictions and confidence scores
results_df = pd.DataFrame({
    'true_emotion': [id_to_emotion[label] for label in true_labels],
    'predicted_emotion': [id_to_emotion[pred] for pred in predictions],
    'confidence': confidence_scores,
    'correct': [true_labels[i] == predictions[i] for i in range(len(true_labels))]
})

# Calculate average confidence by emotion
avg_conf_by_emotion = results_df.groupby('true_emotion')['confidence'].mean().sort_values(ascending=False)

# Plot
bars = plt.bar(
    avg_conf_by_emotion.index,
    avg_conf_by_emotion.values,
    color='lightgreen'
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.01,
        f'{height:.2f}',
        ha='center', 
        va='bottom',
        rotation=0
    )

plt.xlabel('Emotion')
plt.ylabel('Average Confidence Score')
plt.title('Average Prediction Confidence by Emotion')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('confidence_by_emotion.png', dpi=300)
print("Confidence distribution chart saved as confidence_by_emotion.png")

print("All visualizations have been generated!")

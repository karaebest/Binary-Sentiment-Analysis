

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

"""Define a function to load the data"""

def load_data(data_dir):
    reviews, labels = [], []
    for sentiment in ["neg", "pos"]:
        sentiment_dir = os.path.join(data_dir, sentiment)
        for review_file in os.listdir(sentiment_dir):
            with open(os.path.join(sentiment_dir, review_file), "r") as f:
                review = f.read()
                review = re.sub("<[^>]*>", "", review)  # Remove HTML tags
                reviews.append(review)
                labels.append(1 if sentiment == "pos" else 0)
    return reviews, labels

"""Load the train and test data"""

train_data_dir = "aclImdb/train"
test_data_dir = "aclImdb/test"

X_train_raw, y_train = load_data(train_data_dir)
X_test_raw, y_test = load_data(test_data_dir)

import torch
from transformers import BertTokenizer, BertForSequenceClassification

"""Load the BERT tokenizer and tokenize the train and test data"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", )

X_train_tokenized = [tokenizer.encode(review, max_length=512, truncation=True, padding='max_length') for review in X_train_raw]
X_test_tokenized = [tokenizer.encode(review, max_length=512, truncation=True, padding='max_length') for review in X_test_raw]

"""Convert the tokenized train and test data to PyTorch tensors"""

X_train_tensors = torch.tensor(X_train_tokenized)
X_test_tensors = torch.tensor(X_test_tokenized)
y_train_tensors = torch.tensor(y_train)
y_test_tensors = torch.tensor(y_test)

"""Create the DataLoader for the train and test data"""

from torch.utils.data import TensorDataset, DataLoader

batch_size = 8

train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
test_dataset = TensorDataset(X_test_tensors, y_test_tensors)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""Load the pre-trained BERT model for sequence classification"""

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", output_attentions=True)
model.cuda()  # Move the model to GPU if available

"""Set up the training configuration"""

from transformers import AdamW, get_linear_schedule_with_warmup

epochs = 3
total_steps = len(train_dataloader) * epochs

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

"""Define helper functions for training and evaluation"""

import time
import datetime
from tqdm import tqdm

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()

    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        model.zero_grad()

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    for batch in tqdm(dataloader, desc="Evaluation"):
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs.loss, outputs.logits

        total_loss += loss.item()

        _, preds = torch.max(logits, dim=1)
        total_correct += torch.sum(preds == labels).item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

"""Train and evaluate the BERT model"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    print(f"Training loss: {train_loss:.4f}")

    eval_loss, eval_acc = evaluate(model, test_dataloader, device)
    print(f"Test loss: {eval_loss:.4f}")
    print(f"Test accuracy: {eval_acc:.4f}")

    print()

"""# Attention Matrices

"""

# Import statements
from operator import add
import matplotlib.pyplot as plt
import seaborn as sns

# Incorrect classification

input_text = "it is not as awesome as everyone says"
true_label = 0

# Tokenize the input text
input_tokens = tokenizer.encode(input_text, max_length=512, truncation=True)

# Convert the input tokens to a PyTorch tensor
input_tensor = torch.tensor(input_tokens).unsqueeze(0)

# Move the input tensor to the GPU if available
input_tensor = input_tensor.to(device)

# Get the model outputs for the input tensor
outputs = model(input_tensor)

# Get the logits from the model outputs
logits = outputs.logits

# Get the predicted label
_, pred_label = torch.max(logits, dim=1)

# Print the input text, true label, and predicted label
print('Input Text:', input_text)
print('True Label:', true_label)
print('Predicted Label:', pred_label.item())

# Check if the attentions attribute is not None
if outputs.attentions is not None:

  # Get the attention matrix from the model outputs
  attention_matrix = outputs.attentions[-1].squeeze(0)

  class_attentions = []

  # Get the attention weights for a specific head
  attention_weights = attention_matrix[9]
  class_attention = attention_weights[0, :] # Get the class tokens
  class_attentions.append(class_attention.detach().cpu().numpy())

  # Create the labels
  labels=[]
  for i in range(len(class_attention)):
    labels.append(tokenizer.convert_ids_to_tokens(input_tokens[i]))

  # Plot heatmap
  plt.figure(figsize=(12, 4))
  sns.heatmap(np.array(class_attentions), annot=True,xticklabels=labels, cmap='mako', linewidths=0.5)
  plt.title("Attention Matrix for a Incorrectly Predicted Document")
  plt.xlabel("Word tokens")
  plt.ylabel("Class tokens of each attention head")
  plt.show()

# Correct classification

input_text = "I really loved this one"
true_label = 1

# Tokenize the input text
input_tokens = tokenizer.encode(input_text, max_length=512, truncation=True)

# Convert the input tokens to a PyTorch tensor
input_tensor = torch.tensor(input_tokens).unsqueeze(0)

# Move the input tensor to the GPU if available
input_tensor = input_tensor.to(device)

# Get the model outputs for the input tensor
outputs = model(input_tensor)

# Get the logits from the model outputs
logits = outputs.logits

# Get the predicted label
_, pred_label = torch.max(logits, dim=1)

# Print the input text, true label, and predicted label
print('Input Text:', input_text)
print('True Label:', true_label)
print('Predicted Label:', pred_label.item())

# Check if the attentions attribute is not None
if outputs.attentions is not None:

  # Get the attention matrix from the model outputs
  attention_matrix = outputs.attentions[-1].squeeze(0)

  class_attentions = []

  # Get the attention weights for a specific head
  attention_weights = attention_matrix[10]
  class_attention = attention_weights[:, 0] # Get the class tokens
  class_attentions.append(class_attention.detach().cpu().numpy())

  # Create the labels
  labels=[]
  for i in range(len(class_attention)):
    labels.append(tokenizer.convert_ids_to_tokens(input_tokens[i]))

  # Plot heatmap
  plt.figure(figsize=(12, 4))
  sns.heatmap(np.array(class_attentions), xticklabels=labels, annot=True, cmap='mako', linewidths=0.5)
  plt.title("Attention Matrix for an Correctly Predicted Document")
  plt.xlabel("Word tokens")
  plt.ylabel("Class tokens of each attention head")
  plt.show()
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Preprocess data
def preprocess_data(data_df):
    # Split data
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["value"], padding="max_length", truncation=True, max_length=128)
        # Use the "flag" column as labels, converting boolean to int (0 or 1)
        tokenized["labels"] = [int(flag) for flag in examples["flag"]]
        return tokenized
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    return train_dataset, test_dataset


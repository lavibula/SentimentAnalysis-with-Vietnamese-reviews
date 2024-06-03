import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer,Trainer, TrainingArguments, AutoConfig, AutoModelForSequenceClassification,AutoModel
from data_preprocess import TextProcessor
from torch.utils.data import Dataset

class DataHandler:
    def __init__(self, model_name, preprocessor):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.preprocessor = preprocessor
        self.sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        self.topic_mapping = {0: "LECTURER", 1: "TRAINING_PROGRAM", 2: "FACILITY", 3: "OTHERS"}

    def load_data(self, train_path, valid_path, test_path):
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        return train_df, valid_df, test_df
    
    def preprocess_data(self, df):
        # Ensure 'sentence_dropped' column contains only strings
        df['sentence_dropped'] = df['sentence_dropped'].astype(str)
        df['sentence_dropped'] = df['sentence_dropped'].apply(self.preprocessor.preprocess)
        df['combined'] = df.apply(lambda row: f"{self.sentiment_mapping[row['sentiment']]}_{self.topic_mapping[row['topic']]}", axis=1)
        return df


    def encode_labels(self, df):
        combined_labels = pd.get_dummies(df['combined']).values
        return combined_labels

    def tokenize_data(self, df):
        return self.tokenizer(
            df['sentence_dropped'].tolist(),
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )

    def prepare_datasets(self, train_df, valid_df, test_df):
        train_encodings = self.tokenize_data(train_df)
        valid_encodings = self.tokenize_data(valid_df)
        test_encodings = self.tokenize_data(test_df)

        y_train_combined = self.encode_labels(train_df)
        y_valid_combined = self.encode_labels(valid_df)
        y_test_combined = self.encode_labels(test_df)

        return train_encodings, valid_encodings, test_encodings, y_train_combined, y_valid_combined, y_test_combined

    class CustomDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

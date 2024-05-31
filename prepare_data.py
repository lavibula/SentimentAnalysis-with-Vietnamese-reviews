from torch.utils.data import Dataset
import torch
import pandas as pd
import logging

TRAIN_DATA_PATH = "Data_preprocessed/Train.csv"
VAL_DATA_PATH = "Data_preprocessed/Validation.csv"

class SentimentDataset(Dataset):
    def __init__(self, sentences, sentiment, topic, tokenizer = None, max_length = 128, padding = "max_length", truncation = True, tokenized = False):
        self.logger = logging.getLogger("SentimentDataset")
        self.sentiment = sentiment
        self.topic = topic
        self.num_sentiments = len(set(sentiment))

        # Tokenize the input sentences beforehand for faster loading
        if tokenized:
            self.tokenized = sentences
        elif tokenizer is None:
            raise ValueError(f"When {tokenized = }, tokenizer must not be None")
        else:
            self.logger.info("Tokenizing...")
            self.tokenized = tokenizer(sentences, padding = padding, max_length = max_length, truncation = truncation)
    
    def __len__(self):
        return len(self.topic)
    
    def __getitem__(self, idx):
        token = {key: torch.tensor(val[idx]) for key, val in self.tokenized.items()}
        topic = self.topic[idx]
        sentiment = self.sentiment[idx]
        gt = self.num_sentiments * topic + sentiment # Merge topic and sentiment into one label
        token['label'] = torch.tensor(gt)
        return token
    
def load_data(data_path, tokenizer, tokenized = False, max_length = 256, padding = "max_length", truncation = True):
    df = pd.read_csv(data_path).dropna()
    sentences = df["sentence_dropped"].tolist()
    sentiment = df["sentiment"].tolist()
    topic = df["topic"].tolist()
    return SentimentDataset(sentences, sentiment, topic, tokenizer = tokenizer, max_length = max_length, padding = padding, truncation = truncation, tokenized = tokenized)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    train_dataset = load_data(TRAIN_DATA_PATH, tokenizer)
    print(train_dataset[0])
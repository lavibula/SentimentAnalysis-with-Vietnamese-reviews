import os
import yaml
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

class SentimentAnalysisModel:
    def __init__(self, config_path, wandb_token, hf_token, epochs):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.save_pretrained_path = config['save_pretrained_path']
        self.tokenizer_save_path = config['tokenizer_save_path']
        self.hub_model_name = config['hub_model_name']
        self.project_name = config['project_name']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.warmup_steps = config['warmup_steps']
        self.weight_decay = config['weight_decay']
        self.logging_dir = config['logging_dir']
        self.logging_steps = config['logging_steps']
        self.evaluation_strategy = config['evaluation_strategy']
        self.eval_steps = config['eval_steps']
        self.save_steps = config['save_steps']
        self.save_total_limit = config['save_total_limit']
        self.dataloader_num_workers = config['dataloader_num_workers']
        self.dataloader_prefetch_factor = config['dataloader_prefetch_factor']
        self.report_to = config['report_to']
        self.run_name = config['run_name']
        self.load_best_model_at_end = config['load_best_model_at_end']
        
        self.wandb_token = wandb_token
        self.hf_token = hf_token
        self.epochs = epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
        
    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        
        # Extract sentiment and topic from combined labels
        preds_sentiments = preds // 4
        preds_topics = preds % 4
        labels_sentiments = labels // 4
        labels_topics = labels % 4
        
        if len(labels_sentiments.shape) > 1 and labels_sentiments.shape[1] > 1:
            labels_sentiments = labels_sentiments.argmax(axis=1)
        if len(labels_topics.shape) > 1 and labels_topics.shape[1] > 1:
            labels_topics = labels_topics.argmax(axis=1)
        
        # Debug prints to check shapes and contents of the arrays
        print(f"Preds sentiments: {preds_sentiments}")
        print(f"Preds topics: {preds_topics}")
        print(f"Labels sentiments: {labels_sentiments}")
        print(f"Labels topics: {labels_topics}")

        # Ensure all arrays are 1-dimensional
        preds_sentiments = preds_sentiments.flatten()
        preds_topics = preds_topics.flatten()
        labels_sentiments = labels_sentiments.flatten()
        labels_topics = labels_topics.flatten()
        
        # Compute metrics for sentiment classification
        precision_sentiment, recall_sentiment, f1_sentiment, _ = precision_recall_fscore_support(labels_sentiments, preds_sentiments, average='weighted')
        acc_sentiment = accuracy_score(labels_sentiments, preds_sentiments)
        
        # Compute metrics for topic classification
        precision_topic, recall_topic, f1_topic, _ = precision_recall_fscore_support(labels_topics, preds_topics, average='weighted')
        acc_topic = accuracy_score(labels_topics, preds_topics)
        
        return {
            'accuracy_sentiment': acc_sentiment,
            'f1_sentiment': f1_sentiment,
            'precision_sentiment': precision_sentiment,
            'recall_sentiment': recall_sentiment,
            'accuracy_topic': acc_topic,
            'f1_topic': f1_topic,
            'precision_topic': precision_topic,
            'recall_topic': recall_topic
        }
    
    def debug_data(self, dataset):
        print("Debugging dataset:")
        for i in range(3):  # Print first 3 samples
            item = dataset[i]
            print(f"Sample {i}:")
            print(f"  Keys: {list(item.keys())}")
            print(f"  Labels: {item['labels']}")
            print()

    def train(self, train_dataset, valid_dataset, test_dataset):
        os.environ["WANDB_API_KEY"] = self.wandb_token
        wandb.init(project=self.project_name, config={
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        })

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=16,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            report_to=self.report_to,
            run_name=self.run_name,
            load_best_model_at_end=self.load_best_model_at_end
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Debug the datasets before training
        self.debug_data(train_dataset)
        self.debug_data(valid_dataset)
        self.debug_data(test_dataset)

        trainer.train()

        self.model.save_pretrained(self.save_pretrained_path)
        self.tokenizer.save_pretrained(self.tokenizer_save_path)

        self.model.push_to_hub(self.hub_model_name, use_auth_token=self.hf_token)
        self.tokenizer.push_to_hub(self.hub_model_name, use_auth_token=self.hf_token)

    def get_prediction(self, sentence):
        self.model.eval()
        if not isinstance(sentence, str):
            raise ValueError("Input sentence must be a string.")
        
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Define mappings
        sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        topic_mapping = {0: "LECTURER", 1: "TRAINING_PROGRAM", 2: "FACILITY", 3: "OTHERS"}

        # Extract sentiment and topic from combined label
        sentiment_index = predicted_label // 4
        topic_index = predicted_label % 4

        # Get the sentiment and topic strings
        sentiment = sentiment_mapping[sentiment_index]
        topic = topic_mapping[topic_index]

        return {
              "Topic": topic,
              "Sentiment": sentiment }

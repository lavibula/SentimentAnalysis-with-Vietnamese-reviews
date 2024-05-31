from custom_transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
from prepare_data import load_data

import logging

import os
os.environ["WANDB_PROJECT"]="VietnameseSentimentAnalysis"
#import wandb
#wandb.login(key="3f1b76982c335dfa2a93d09ae7cf4b34a640bc60")

logging.basicConfig(level = logging.INFO)

TRAIN_DATA_PATH = "Data_preprocessed/Train.csv"
VAL_DATA_PATH = "Data_preprocessed/Validation.csv"
PRETRAINED_MODEL = "vinai/phobert-base"
NUM_LABELS = 16

def compute_metrics(pred, metric):
    logits, labels = pred
    predictions = logits.argmax(-1)
    return metric.compute(predictions = predictions, references = labels)

def main():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    config = RobertaConfig(problem_type = "single_label_classification", num_labels = NUM_LABELS, use_cls = False, classifier_input_length = 256*768, hidden_size=120)
    classifier = RobertaForSequenceClassification(config).classifier
    model = RobertaForSequenceClassification(config).from_pretrained("vinai/phobert-base")
    model.classifier = classifier
    model.num_labels = NUM_LABELS
    train_dataset = load_data(TRAIN_DATA_PATH, tokenizer)
    val_dataset = load_data(VAL_DATA_PATH, tokenizer)

    metric = evaluate.load("accuracy")
    compute_metric = lambda pred: compute_metrics(pred, metric)
    
    training_args = TrainingArguments(
        output_dir = "output",
        num_train_epochs = 30,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        warmup_steps = 500,
        weight_decay = 0.01,
        logging_dir = "logs",
        logging_steps = 100,
        eval_strategy = "steps",
        eval_steps = 1000,
        save_steps = 1000,
        save_total_limit = 3,
        dataloader_num_workers = 2,
        dataloader_prefetch_factor = 2,
        #report_to="wandb",
        #run_name="phobert-cls"
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metric,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
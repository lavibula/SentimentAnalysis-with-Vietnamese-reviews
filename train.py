from custom_transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
from prepare_data import load_data

import logging

import os

logging.basicConfig(level = logging.INFO)

def compute_metrics(pred, metrics):
    metric_names = [m for m in metrics.keys()]
    metric_args = [metrics[m] for m in metric_names]
    for idx, args in enumerate(metric_args):
        if args is None:
            metric_args[idx] = {}
    metrics = [evaluate.load(m) for m in metric_names]

    logits, labels = pred
    predictions = logits.argmax(-1)
    return {name: metric.compute(predictions=predictions, references=labels, **args) for name, metric, args in zip(metric_names, metrics, metric_args)}

def get_config(path):
    """Load the config from a yml file"""
    import yaml
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    ymlconfig = get_config("./config.yml")

    import wandb
    wandb.login(key=ymlconfig['wandb']['key'])
    os.environ["WANDB_PROJECT"]=ymlconfig['wandb']['project']

    TRAIN_DATA_PATH = ymlconfig['data']['train_path']
    VAL_DATA_PATH = ymlconfig['data']['val_path']
    NUM_LABELS = ymlconfig['data']['num_labels']

    MODEL_CONFIG = ymlconfig['model']
    PRETRAINED_MODEL = MODEL_CONFIG['pretrain']

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    config = RobertaConfig(**MODEL_CONFIG['config'])
    classifier = RobertaForSequenceClassification(config).classifier
    model = RobertaForSequenceClassification(config).from_pretrained(PRETRAINED_MODEL)
    model.classifier = classifier
    model.num_labels = NUM_LABELS
    train_dataset = load_data(TRAIN_DATA_PATH, tokenizer)
    val_dataset = load_data(VAL_DATA_PATH, tokenizer)

    compute_metric = lambda pred: compute_metrics(pred, ymlconfig['eval']['metrics'])
    
    training_args = TrainingArguments(**ymlconfig['train'])
    
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
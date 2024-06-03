from ktrnh.src.data.data_preprocess import TextPreprocessor
from ktrnh.src.data.data_handler import DataHandler
from ktrnh.src.model.model import SentimentAnalysisModel
import argparse

def main(args):
    preprocessor = TextPreprocessor()
    data_handler = DataHandler(model_name="vinai/phobert-base", preprocessor=preprocessor)
    train_df, valid_df, test_df = data_handler.load_data(args.train_path, args.valid_path, args.test_path)
    
    # Preprocess the data
    train_df = data_handler.preprocess_data(train_df)
    valid_df = data_handler.preprocess_data(valid_df)
    test_df = data_handler.preprocess_data(test_df)
    
    train_encodings, valid_encodings, test_encodings, y_train_combined, y_valid_combined, y_test_combined = data_handler.prepare_datasets(train_df, valid_df, test_df)

    train_dataset = DataHandler.CustomDataset(train_encodings, y_train_combined)
    valid_dataset = DataHandler.CustomDataset(valid_encodings, y_valid_combined)
    test_dataset = DataHandler.CustomDataset(test_encodings, y_test_combined)

    sentiment_model = SentimentAnalysisModel(
        config_path=args.config,
        wandb_token=args.wandb_token,
        hf_token=args.hf_token,
        epochs=args.epochs
    )

    sentiment_model.train(train_dataset, valid_dataset, test_dataset)

    sample_sentence = "giáo viên rất vui tính"
    predicted_label = sentiment_model.get_prediction(sample_sentence)
    print(f"Predicted sentiment for '{sample_sentence}': {predicted_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--wandb_token", type=str, required=True, help="WandB API token")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--valid_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset")
    args = parser.parse_args()
    main(args)

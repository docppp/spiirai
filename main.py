import argparse
import dill as pickle
import pandas as pd

import config


def train_model(train_file_path, model_save_path):
    from ai import AiEngine
    from keras.models import save_model
    ai_engine = AiEngine(train_file_path)
    ai_engine.train()

    save_model(ai_engine.model, model_save_path + '.h5')
    del ai_engine.model
    with open(model_save_path + '.pickle', 'wb') as f:
        pickle.dump(ai_engine, f)
    print("Trained model saved successfully.")


def use_model(model_file_path, new_data):
    from keras.models import load_model
    new_data_df = pd.read_csv(new_data, sep=";", encoding="ansi")
    with open(model_file_path + '.pickle', 'rb') as f:
        ai_engine = pickle.load(f)
    ai_engine.model = load_model(model_file_path + '.h5')
    ai_engine.guess(new_data_df)


def main():
    parser = argparse.ArgumentParser(description='Spiir AI Engine')
    parser.add_argument('option', choices=['train', 'guess'],
                        help='Choose "train" to train the model or "guess" to use the model')
    parser.add_argument('--train_data', help='Path to the training data file')
    parser.add_argument('--model_file', required=True, help='Path to the model file w/o extension (save or use)')
    parser.add_argument('--new_data', help='New data for prediction')

    args = parser.parse_args()

    config.load_config()

    if args.option == 'train':
        if not args.train_data or not args.model_file:
            print('Please provide both training file path and save model path')
            return
        train_model(args.train_data, args.model_file)

    elif args.option == 'guess':
        if not args.new_data or not args.model_file:
            print('Please provide both model file path and new data for prediction')
            return
        use_model(args.model_file, args.new_data)


if __name__ == '__main__':
    main()

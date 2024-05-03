import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from data import Data
from config import config


class AiEngine:
    def __init__(self, train_file_path):
        self.train_data = Data(train_file_path)
        self.scaler = StandardScaler()
        self.model = Sequential()
        input_layer_size = config['model']['words_limit'] + 1 + 5  # + 1 (for unknown words) + 5 (input columns)
        self.model.add(Dense(config['model']['layer1_size'], input_dim=input_layer_size, activation='relu'))
        self.model.add(Dense(config['model']['layer2_size'], activation='relu'))
        self.model.add(Dense(config['model']['layer3_size'], activation='relu'))
        self.model.add(Dense(100, activation='softmax'))  # number of categories

    def train(self):
        self.train_data.preprocess()
        df = self.train_data.df

        # df[['AmountInteger']] = self.scaler.fit_transform(df[['AmountInteger']])

        # Prepare inputs and outputs
        x = df[['DayOfYear', 'DayOfMonth', 'DayOfWeek', 'AmountInteger', 'AmountDecimal']].values
        x = np.hstack([x, np.stack(df['Description'].values)])
        y = df['CategoryId'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                       epochs=config['training']['epochs'],
                       batch_size=config['training']['batch_size'],
                       validation_split=0.2)

        loss, accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {accuracy * 100:.2f}%")

    def guess(self, row: pd.DataFrame):
        row = self.train_data.process_rows(row)
        # row[['AmountInteger']] = self.scaler.transform(row[['AmountInteger']])
        x = row[['DayOfYear', 'DayOfMonth', 'DayOfWeek', 'AmountInteger', 'AmountDecimal']].values
        print(x)
        x = np.hstack([x, np.stack(row['Description'].values)])
        predictions = self.model.predict(x)
        predicted_category = np.argmax(predictions, axis=1)
        top_n_predictions = np.argsort(-predictions, axis=1)[:, :3]
        top_probabilities = predictions[0][top_n_predictions]
        print(f"Category: {top_n_predictions}, Probability: {top_probabilities}")

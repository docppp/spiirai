import pandas as pd

from config import config

pd.options.mode.copy_on_write = True


class Data:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep=";", encoding="ansi")
        self.most_common_words = []
        self.words_set = ()
        self.words_index = {}
        self.most_common_words_limit = config['model']['words_limit']

        self.data_pipeline = lambda df: (
            df.pipe(self._drop_hidden)
            .pipe(self._drop_extraordinary)
            .pipe(self._drop_unused_column)
            .pipe(self._process_date_column)
            .pipe(self._split_amount_column)
            .pipe(self._process_category_id_column)
        )

    def preprocess(self):
        self.df = self.data_pipeline(self.df)

        all_words_series = pd.Series(' '.join(self.df["Description"]).split())
        self.most_common_words = all_words_series.value_counts().head(self.most_common_words_limit).index.tolist()
        self.words_set = set(self.most_common_words)
        self.words_index = {word: i for i, word in enumerate(self.most_common_words)}

        self.df["Description"] = self.df["Description"].apply(self.encode_description)

    def process_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.most_common_words) == 0:
            raise RuntimeError("Data must be preprocessed with training data first")
        df = self.data_pipeline(df)
        df["Description"] = df["Description"].apply(self.encode_description)
        return df

    def encode_description(self, description):
        encoded = [0] * self.most_common_words_limit
        words = description.split()
        not_found_count = 0

        for word in words:
            if word in self.words_set:
                encoded[self.words_index[word]] = 1
            else:
                not_found_count -= 1

        encoded.append(not_found_count)
        return encoded

    @staticmethod
    def _drop_hidden(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["MainCategoryName"] != "Hide"]
        return df

    @staticmethod
    def _drop_extraordinary(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Extraordinary"] != "Yes"]
        df.drop(columns=["Extraordinary"], inplace=True)
        return df

    @staticmethod
    def _drop_unused_column(df: pd.DataFrame) -> pd.DataFrame:
        columns_to_remove = [
            "Id",
            "AccountId",
            "AccountName",
            "AccountType",
            "OriginalDescription",
            "MainCategoryId",
            "MainCategoryName",
            "CategoryName",
            "CategoryType",
            "ExpenseType",
            "Balance",
            "CounterEntryId",
            "Comment",
            "Tags",
            "SplitGroupId",
            "Currency",
            "OriginalAmount",
            "OriginalCurrency"
        ]
        df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        return df

    @staticmethod
    def _process_date_column(df: pd.DataFrame) -> pd.DataFrame:
        df['Date'] = df['CustomDate'].combine_first(df['Date'])
        df.drop(columns=['CustomDate'], inplace=True)

        df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')
        df['DayOfYear'] = df["Date"].dt.dayofyear / 365
        df['DayOfMonth'] = df["Date"].dt.day / 31
        df['DayOfWeek'] = df["Date"].dt.dayofweek / 7
        df.drop(columns=["Date"], inplace=True)

        return df

    @staticmethod
    def _split_amount_column(df: pd.DataFrame) -> pd.DataFrame:
        df['AmountInteger'] = df["Amount"].apply(lambda x: int(str(x).split(',')[0]))
        df['AmountDecimal'] = df["Amount"].apply(lambda x: int(str(x).split(',')[1]) / 100 if ',' in str(x) else 0)
        df.drop(columns=["Amount"], inplace=True)
        return df

    @staticmethod
    def _process_category_id_column(df: pd.DataFrame) -> pd.DataFrame:
        df['CategoryId'] = df['CategoryId'] - 100
        return df

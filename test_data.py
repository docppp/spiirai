import unittest
from copy import deepcopy
from data import Data


class TestData(unittest.TestCase):

    test_data = Data("test_data.txt", 9)

    def test_drop_hidden(self):
        df = Data._drop_hidden(self.test_data.df)
        self.assertTrue((df['MainCategoryName'] != 'Hide').all())

    def test_process_date(self):
        df = Data._process_date_column(self.test_data.df)
        self.assertAlmostEqual(df.iloc[0]["DayOfWeek"], 0 / 7)
        self.assertAlmostEqual(df.iloc[0]["DayOfMonth"], 18 / 31)
        self.assertAlmostEqual(df.iloc[0]["DayOfYear"], 78 / 365)

        self.assertAlmostEqual(df.iloc[-1]["DayOfWeek"], 1 / 7)
        self.assertAlmostEqual(df.iloc[-1]["DayOfMonth"], 2 / 31)
        self.assertAlmostEqual(df.iloc[-1]["DayOfYear"], 93 / 365)

    def test_split_amount(self):
        df = Data._split_amount_column(self.test_data.df)
        self.assertEqual(df.iloc[0]["AmountInteger"], -10)
        self.assertEqual(df.iloc[0]["AmountDecimal"], 0)

        self.assertEqual(df.iloc[-2]["AmountInteger"], -40)
        self.assertEqual(df.iloc[-2]["AmountDecimal"], 0.54)

    def test_description(self):
        data = deepcopy(self.test_data)
        data.preprocess()
        self.assertEqual(data.most_common_words[0], "CITY")
        self.assertEqual(data.df.iloc[9]["Description"], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

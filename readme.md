## Spiir Ai Categorizer

This project aims to develop an AI model capable of categorizing expenses in a home budget dataset.
It involves data preprocessing, model training, and prediction functionalities.

## Data needed for training
To acquire data, goto your spiir profile at https://mine.spiir.dk/profil/eksport and export all history in csv format.
This file will be used for training, the longer history, the better.
Due to obvious reasons, this file is not included in the repo.

## Usage
`python main.py train --train_data <training_data_file_path> --model_file <model_save_path>
`

This command will produce 2 files `model_save_path.pickle` with dumped ai engine and `model_save_path.h5` with actual model.
Total model accuracy will be printed (for my dataset it is ~75-80%)

`python main.py guess --model_file <model_file_path> --new_data <new_data_file_path>
`
This command will use model generated in previous step to guess category of new transactions,
which should have exact same format as data exported from Spiir. 
Note that the `.h5` or `.pickle` extension should not be included - they are added automatically.

## Output representation
For each line provided in new data set, 6 numbers are printed. 
Eg for model trained on my personal data, guessing `new_data.csv`:

`Category: [[88 40 13]], Probability: [[0.618317   0.35009715 0.01039907]]`

This line says that the most probable category of one-line input is 188 (not 88!) with chance 61%.

During processing 100 is subtracted from all CategoryIds to reduce number of output nodes.
This project is developed based on my personal data and I do not use all categories in Spiir,
therefore I am unable to create proper map CategoryName <-> CategoryId. 

## Specification

### Data preprocessing
Data class is designed to handle data preprocessing for training an AI model used for categorizing expenses in a home budget. 
It involves several steps of data cleaning, transformation, and encoding to prepare the data for fitting into a Keras model.

It finds 999 (configurable) most common words in all descriptions and encodes them into one vector of 1s and 0s.
Additional value is inserted in this vector that represent how many uncommon words are in description.

In addition to the categorical data in the "Description", there are numerical features such as "Amount" and "Date" for better expense categorization.

The "Amount" column represents the monetary value of each expense.
During preprocessing, the amount is split into integer and decimal parts to capture the whole and fractional components accurately.
The integer part is scaled using standard scaling to bring all numerical features to a similar scale, enhancing model performance.
The processed "Amount" is then included as input alongside other features for training the model.

The "Date" column indicates the date when each expense occurred.
To extract meaningful features from dates, the day of year, day of month, and day of week are derived.
These features provide temporal context to the expenses and help the model learn patterns related to spending behavior over different time periods.

### Ai Engine
Uses Keras neural network model. It utilizes the Data class for preprocessing the training data and incoming data for predictions.
It compiles and trains the neural network model using the specified loss function, optimizer, and metrics.
Also, evaluates the model's performance on the test set and prints the test accuracy.

### Deps
    pandas
    numpy
    scikit-learn
    Keras
    dill

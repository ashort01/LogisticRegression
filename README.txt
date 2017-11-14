The following files are included:

----------ProcessAudio.py------------
- This is a python script to load and process the data. 
- It has many depedencies which can be found in the import section of the file.
- This was designed in a Windows environment with python 3.6
- It requires a /data/genres folder with all the genres, and a data/validation folder with the test data
- We used this script to load and preprocess our audio data and load features and classes into csvs
  for modeling.
- This script does not do any Machine learning, just data processing and feature extraction.

----------LogisticRegression.py------------
- This script primarily uses the sklearn library.
- Reads data from csvs (from ProcessAudio.py) into arrays to train and test a LR model
- It also does normalization on the data.
- outputs a confusion matrix, accuracy score, and results.csv (for kaggle).

----------------MLP.py--------------------------
- This script primarily uses the sklearn library.
- Reads data from csvs (from ProcessAudio.py) into arrays to train and test a MLP Neural Network model
- It also does normalization on the data.
- outputs a confusion matrix, accuracy score, and results.csv (for kaggle).


---------CS 529 - Project 3.pdf-----------
- The report for the project.


Data folders are not included due to size limitations.


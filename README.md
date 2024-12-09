# Twitter Sentiment Analysis
## Project Overview
This project aims to build a sentiment analysis model for classifying tweets into two categories: suspicious and non-suspicious. We explore different machine learning techniques, including Support Vector Machine (SVM), Long Short-Term Memory (LSTM), and BERT for the analysis of text data. The project involves data preprocessing, model training, evaluation, and result comparison.

Clone this repository to your local machine:

 bash 
 ``` git clone https://github.com/kavya-daya/TwitterSentimentAnalysis.git ```

Install the required libraries and dependencies:

bash
```pip install -r requirements.txt ```
Dataset
The dataset used for training and testing the models consists of tweets labeled as either suspicious or non-suspicious. The dataset is available for download from the following sources:

## Kaggle Suspicious Tweets Dataset
Original Dataset
- The dataset is preprocessed to remove unwanted patterns like mentions, URLs, and other irrelevant characters. Tokenization, stemming, and stopword removal are performed to prepare the text data for model training.

## Models Implemented
- This project includes the following models for sentiment classification:

Support Vector Machine (SVM):
- A traditional machine learning model for text classification.
Long Short-Term Memory (LSTM):
- A deep learning model used to process and analyze sequential text data.
BERT (Bidirectional Encoder Representations from Transformers):
- A transformer-based model that has shown state-of-the-art performance in various NLP tasks.
Usage
## Preprocessing the Data:

- Clean the tweets by removing mentions, URLs, and other unnecessary characters.
- Tokenize the tweets and convert them into sequences for model input.
- Pad the sequences to ensure uniform input size across all tweets.
## Training the Models:

- Train each model using the preprocessed data.
- The models are trained on different configurations, including hyperparameter tuning for better performance.
## Evaluation:

Evaluate the models on a validation set to check the accuracy, precision, recall, and F1-score.
## Prediction:

Use the trained models to predict labels (suspicious or non-suspicious) for new tweet data.
python
# Example usage of LSTM model
predictions = model.predict(new_x)
## Results
The results from the implementation of the models on the dataset show:

- SVM: Shows decent performance with an accuracy of 85% and precision of 82%.
- LSTM: Achieved an accuracy of 88% and a precision of 87%, outperforming SVM.
- BERT: Outperformed both SVM and LSTM, achieving an accuracy of 92% and precision of 91%.
- However, the paper's original code results in a significantly lower performance with an accuracy of 41%, indicating the need for further improvements and handling of class imbalance.

## Contributing
Contributions are welcome! If you want to improve the model, please fork the repository, make your changes, and submit a pull request.

## How to Contribute:
- Fork the repository.
- Clone your fork to your local machine.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them (git commit -am 'Add new feature').
- Push your changes to your fork (git push origin feature-branch).
- Submit a pull request.

References
* "Comparative Study of Abusive Language Detection Approaches" (https://arxiv.org/pdf/1808.10245v1)
* "Twitter Sentiment Analysis" GitHub Repo (https://github.com/kavya-daya/TwitterSentimentAnalysis)
* Kaggle Suspicious Tweets Dataset (https://www.kaggle.com/datasets/syedabbasraza/suspicious-tweets?resource=download)

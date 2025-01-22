# BLSTM-Sentiment-Analysis-
# BLSTM Sentiment Analysis on E-Commerce Reviews

## Overview
This project implements a Bidirectional Long Short-Term Memory (BLSTM) model for sentiment analysis on e-commerce reviews. The reviews are sourced from three major platforms: Zepto, Jiomart, and Blinkit. The sentiment analysis categorizes reviews into three classes: `positive`, `neutral`, and `negative` based on the rating provided by users.

---

## Features
- Preprocessing of text data, including tokenization and padding.
- Multi-class classification of reviews using BLSTM.
- Early stopping for efficient training.
- Visualization of training history (accuracy and loss).
- Predictive functionality for individual reviews.

---

## Dataset
The dataset is assumed to be a CSV file named `reviews.csv` with the following columns:
- `review`: The text of the review.
- `rating`: A numerical rating associated with the review.

---

## Dependencies
Make sure to install the following Python libraries:
- pandas
- scikit-learn
- tensorflow
- keras
- plotly

Install the required libraries using:
```bash
pip install pandas scikit-learn tensorflow keras plotly
```

---

## Code Explanation

### 1. Data Preprocessing
- **Label Encoding:** The `rating` is converted into categorical labels (`positive`, `neutral`, `negative`).
- **Tokenization:** The `review` column is tokenized and converted into sequences of integers.
- **Padding:** The sequences are padded to a maximum length of 100.

### 2. Train-Test Split
The dataset is split as follows:
- 70% for training.
- 15% for validation.
- 15% for testing.

### 3. Model Architecture
- **Embedding Layer:** Converts tokenized sequences into dense vectors.
- **SpatialDropout1D:** Adds dropout regularization.
- **Bidirectional LSTM:** Captures context in both forward and backward directions.
- **Dense Layer:** Outputs probabilities for each class using a softmax activation function.

### 4. Training
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metric: Accuracy
- Early stopping is implemented to prevent overfitting.

### 5. Evaluation
The model is evaluated on the test set, and a classification report is generated to display precision, recall, and F1-score for each class.

### 6. Visualization
The training history (accuracy and loss) is visualized using Plotly.

### 7. Prediction
A function is provided to predict the sentiment of a new review and display the confidence score.

---

## Example Usage
### Training
Run the script to preprocess the data, train the model, and evaluate its performance.

### Predicting Sentiment
Use the `predict_review` function to predict the sentiment of a given review:
```python
example_review = "I love the fast delivery and great service!"
print(predict_review(example_review, model, tokenizer, label_encoder))
```

---

## Results
- Test Accuracy: Achieved after training on the provided dataset.
- Classification Report:
  Displays precision, recall, and F1-score for `positive`, `neutral`, and `negative` classes.

---

## Repository Structure
```
.
├── reviews.csv                # Dataset file
├── sentiment_analysis.py      # Main script for training and evaluation
├── README.md                  # Documentation
└── requirements.txt           # Dependencies
```

---

## To Do
- Improve the preprocessing pipeline by incorporating techniques like stemming and lemmatization.
- Experiment with hyperparameter tuning for better performance.
- Add support for additional datasets or languages.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- TensorFlow and Keras for providing tools to build and train neural networks.
- Plotly for interactive visualizations.
- scikit-learn for preprocessing and evaluation utilities.


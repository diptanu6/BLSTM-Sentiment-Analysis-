from keras.models import load_model # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle

from sklearn.calibration import LabelEncoder

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def predict_review(review, model, tokenizer, label_encoder, maxlen=100):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded_sequence, verbose=0)

    class_idx = prediction.argmax(axis=1)[0]
    label = label_encoder.inverse_transform([class_idx])[0]

    confidence = prediction[0][class_idx]

    return f"Sentiment: {label}, Confidence: {confidence:.2f}"

def main():
    model = load_model('sentiment_model.h5')
    tokenizer = load_tokenizer('tokenizer.pkl')

    # Replace with your label encoder logic
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    label_encoder = LabelEncoder()
    label_encoder.classes_ = list(label_mapping.values())

    reviews = [
        "I love the fast delivery and great service!",
        "Terrible Service. Would not order",
        "App is okay.",
        "Delivery can be faster."
    ]

    for review in reviews:
        print(review)
        print(predict_review(review, model, tokenizer, label_encoder))

if __name__ == "__main__":
    main()

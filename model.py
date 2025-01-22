

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense # type: ignore
from keras.callbacks import EarlyStopping # type: ignore

# Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['rating_label'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'neutral' if x == 3 else 'negative')

    label_encoder = LabelEncoder()
    data['rating_label_encoded'] = label_encoder.fit_transform(data['rating_label'])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['review'].values)

    X = tokenizer.texts_to_sequences(data['review'].values)
    X = pad_sequences(X, maxlen=100)

    y = pd.get_dummies(data['rating_label_encoded']).values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, label_encoder

# Build model
def build_model(vocabulary_size, num_classes, embed_dim=128, lstm_out=196):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim=embed_dim))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Train model
def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# Save model and tokenizer
def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)

# Main function
def main():
    filepath = 'C:\\Users\\USER\\OneDrive\\Documents\\Desktop\\PROJECT_25\\reviews.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, label_encoder = load_and_preprocess_data(filepath)

    vocabulary_size = len(tokenizer.word_index) + 1
    num_classes = y_train.shape[1]

    model = build_model(vocabulary_size, num_classes)
    history = train_model(model, X_train, y_train, X_val, y_val)

    save_model_and_tokenizer(model, tokenizer, 'sentiment_model.h5', 'tokenizer.pkl')
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
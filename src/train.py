import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# --- Model Architecture (Moved from model.py to avoid import errors) ---
def build_model(input_shape):
    """Builds an LSTM-based sequential model [cite: 30]"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') # Binary Cross Entropy output [cite: 34]
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Data Preparation ---
def load_and_prep_data(file_path):
    df = pd.read_csv(file_path)
    # Required Features [cite: 17, 19, 20]
    features = ['current_score', 'wickets_remaining', 'balls_remaining', 'over_num', 'ball_num']
    X = df[features].values
    y = df['target'].values
    X = X.reshape((X.shape[0], 1, X.shape[1])) # Reshape for LSTM [cite: 30]
    return train_test_split(X, y, test_size=0.3, random_state=42) # 70% Train split [cite: 34]

# --- Training Execution ---
def train():
    processed_data_path = "data/processed/cleaned_match_data.csv"
    if not os.path.exists(processed_data_path):
        print("Error: Cleaned data not found! Run data_pipeline.py first.")
        return

    X_train, X_test, y_train, y_test = load_and_prep_data(processed_data_path)
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    print("Starting model training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/cricket_win_model.h5") # Save for deployment [cite: 61]
    print("Success! Model saved to models/cricket_win_model.h5")

if __name__ == "__main__":
    train()
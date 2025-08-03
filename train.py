# to train the model
# !pip install joblib

# ================= Import Libraries =================
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ================= Configuration =================
DATA_PATH = './dataset.csv'                             # Path to the dataset
MODEL_DIR = 'model'                                     # Folder to save model and encoder
MODEL_PATH = os.path.join(MODEL_DIR, 'disease_prediction_model.h5')  # Trained model path
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')          # LabelEncoder path
TEST_SIZE = 0.2                                          # 20% data for testing
RANDOM_STATE = 42                                        # Reproducibility
EPOCHS = 50
BATCH_SIZE = 16

# ================= Helper Functions =================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_data(path):
    """Load CSV data into DataFrame"""
    df = pd.read_csv(path)
    return df

def preprocess_symptoms(df):
    """
    Combine all symptom columns into one-hot encoded matrix X.
    - Fills missing values with empty string
    - Maps unique symptoms to indexes
    - Converts each row to a binary vector
    """
    symptom_cols = df.columns[1:]  # Assuming 1st column is Disease
    df['combined_symptoms'] = df[symptom_cols].values.tolist()
    
    # Replace NaNs with empty strings
    df['combined_symptoms'] = df['combined_symptoms'].apply(
        lambda x: [s if pd.notna(s) else '' for s in x]
    )
    
    # Extract all unique symptoms
    all_symptoms = sorted({symptom for sublist in df['combined_symptoms'] for symptom in sublist if symptom})
    
    # Map each symptom to a column index
    symptom_to_index = {symptom: i for i, symptom in enumerate(all_symptoms)}

    # Build one-hot matrix
    X = np.zeros((len(df), len(symptom_to_index)), dtype=np.float32)
    for i, symptom_list in enumerate(df['combined_symptoms']):
        for symptom in symptom_list:
            if symptom in symptom_to_index:
                X[i, symptom_to_index[symptom]] = 1.0

    return X, symptom_to_index

def encode_labels(y_raw):
    """
    Encode disease labels into numeric and one-hot format.
    Returns: one-hot encoded labels and the encoder object
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)  # Converts to numbers
    y_categorical = tf.keras.utils.to_categorical(y_encoded)  # One-hot encode
    return y_categorical, label_encoder

def build_model(input_dim, n_classes):
    """
    Create a simple fully-connected neural network model.
    - Input: symptom one-hot vector
    - Output: probability of each disease class
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')  # Softmax for multiclass classification
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks():
    """
    Define callbacks for:
    - Early stopping to prevent overfitting
    - Reduce learning rate when stuck
    - Save best model
    - TensorBoard logging
    """
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

# ================= Main Training Pipeline =================

def main():
    ensure_dir(MODEL_DIR)

    # Load and preprocess data
    df = load_data(DATA_PATH)
    X, symptom_to_index = preprocess_symptoms(df)

    # Extract target labels (Disease column)
    y_raw = df['Disease'] if 'Disease' in df.columns else df.iloc[:, 0]
    y, label_encoder = encode_labels(y_raw)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )

    # Build and train the model
    model = build_model(X.shape[1], y.shape[1])
    callbacks = get_callbacks()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    # Save model and label encoder for later inference
    model.save(MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"\n✅ Model training complete. Files saved to '{MODEL_DIR}/'")

# Run the training script
if __name__ == '__main__':
    main()

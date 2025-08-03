import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the model and the label encoder
model = load_model('model.h5')

with open('labelencoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('symptom_list.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)  # This should be a list of 131 symptoms in the training order

# User input
user_symptoms = ['itching', 'fatigue', 'weight_loss', 'high_fever', 'mood_swings']

# Convert to binary vector
input_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]
input_vector = np.array(input_vector).reshape(1, -1)

# Optional check
if input_vector.shape[1] != model.input_shape[1]:
    raise ValueError(f"Model expects {model.input_shape[1]} features, but got {input_vector.shape[1]}")

# Predict
pred_encoded = model.predict(input_vector)
predicted_disease = label_encoder.inverse_transform([np.argmax(pred_encoded)])

print("Predicted Disease:", predicted_disease[0])

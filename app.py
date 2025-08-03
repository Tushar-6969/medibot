# main server
from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from chatbot.chatbot import chatbot_handler  
import markdown

app = Flask(__name__)

# Load the trained model and support files
model = load_model('model/model.h5')
with open('model/symptom_list.pkl', 'rb') as f:
    all_symptoms = pickle.load(f)
with open('model/labelencoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', symptoms=all_symptoms, bot_response=None)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    input_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1

    input_array = np.array(input_vector).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    predicted_label = np.argmax(prediction)
    predicted_disease = le.inverse_transform([predicted_label])[0]

    return render_template('result.html', prediction=predicted_disease)

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('query')
    if not user_query:
        return render_template('index.html', symptoms=all_symptoms, bot_response=" Please enter a query.")

    raw_response = chatbot_handler(user_query)
    html_response = markdown.markdown(raw_response)  
    return render_template('index.html', symptoms=all_symptoms, bot_response=html_response)

@app.route('/chatbot', methods=['GET'])
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_query = request.form.get('user_query')
    if not user_query:
        return render_template('chatbot_result.html', user_query="", bot_response=" Please enter a query.")

    raw_response = chatbot_handler(user_query)
    html_response = markdown.markdown(raw_response) 
    return render_template('chatbot_result.html', user_query=user_query, bot_response=html_response)

if __name__ == '__main__':
    app.run(debug=True)

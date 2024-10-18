import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model and tokenizer
def load_resources():
    model_path = 'model/model.h5'
    tokenizer_path = 'model/tokenizer.pickle'

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load the model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

    return tokenizer, model

# Load resources
tokenizer, model = load_resources()

# Function to make predictions
def predict(text):
    if model is None or tokenizer is None:
        return None  # Return if the model or tokenizer failed to load

    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=50)

    # Predict
    prediction = model.predict(padded_sequences)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Streamlit app layout
st.title("Hate Speech Detection")

# Use session state to store the input text
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Text area for input with a unique key
input_text = st.text_area("Enter your text here:", height=200, value=st.session_state.input_text, key="input_text_area")

# Button to make a prediction
if st.button("Predict"):
    if input_text:
        predicted_class = predict(input_text)
        if predicted_class is not None:
            classes = ["Hate Speech", "Offensive Language", "Neither"]
            st.write(f"Prediction: {classes[predicted_class[0]]}")
        else:
            st.error("Prediction failed. Please check the model and tokenizer.")
        
        # Reset the input text in the session state
        st.session_state.input_text = ""  # Clear the input text in session state

    else:
        st.warning("Please enter some text for prediction.")

# Update the text area value from session state
st.session_state.input_text = input_text  # This keeps the latest input for the next rendering

import json
import pickle
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model  # ✅ This line must be here!

# Download tokenizer if not available
nltk.download("punkt")

# Load your model and data
model = load_model("chat_model.h5")
words = pickle.load(open("words.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load intents file
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Clean and preprocess input text
def clean_input(sentence):
    tokens = word_tokenize(sentence.lower())
    bag = [1 if word in tokens else 0 for word in words]
    return np.array([bag])

# Fallback response function
def get_fallback_response():
    """Function to return a fallback response when the model's confidence is below threshold."""
    return "I'm sorry, I didn't quite understand that. Can you please rephrase?"

# Modify predict_intent function with confidence threshold
def predict_class(sentence, threshold=0.75):
    """Predict the intent of a given text input and return the predicted class.
    If the confidence is below the threshold, return the fallback response.
    """
    # Preprocess and tokenize text
    text = clean_input(sentence)
    tokens = word_tokenize(sentence.lower())
    bag = [1 if word in tokens else 0 for word in words]
    
    # Get prediction
    result = model.predict(np.array([bag]))[0]
    max_prob = np.max(result)
    
    # Apply threshold
    if max_prob >= threshold:
        category_index = np.argmax(result)
        tag = label_encoder.inverse_transform([category_index])[0]
        return tag, max_prob
    else:
        # If confidence is below threshold, return fallback response
        return "unknown", max_prob

# Get response based on predicted tag or fallback
def get_response(tag):
    """Get the response from the intent or fallback."""
    if tag == "unknown":
        return get_fallback_response()
    
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])
    
    return "I'm not sure how to respond to that."

# Predict intent and get the response
def predict_intent(sentence):
    """Handles the intent prediction and response fetching."""
    tag, confidence = predict_class(sentence)
    response = get_response(tag)
    return tag, confidence, response

# Chat function
def chat():
    """Function to start the chat with the user."""
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            print("Bot: Goodbye!")
            break
        
        # Predict and get response
        tag, confidence, response = predict_intent(msg)
        
        # Display confidence and response
        print(f"Bot (Confidence: {confidence*100:.2f}%): {response}")

# Run the chatbot
if __name__ == "__main__":
    chat()

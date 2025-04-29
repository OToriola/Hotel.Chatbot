import json
import numpy as np
import nltk
import pickle
import random 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download necessary resources
nltk.download('punkt')

# Load intents
with open("intents.json") as f:
    data = json.load(f)

# Initialize variables
words = []
classes = []
documents = []

# Tokenize and prepare data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern.lower())
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

# Prepare training data
X = []
y = []

for (pattern_tokens, tag) in documents:
    bag = [1 if w in pattern_tokens else 0 for w in words]
    X.append(bag)
    y.append(tag)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = np.array(X)
y = np.array(y_encoded)

# Save for later use
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Build and compile model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# Train the model
model.fit(X, y, epochs=200, batch_size=8)

# Save the model
model.save("chat_model.h5")

# --- New Fallback Response Function ---
def get_fallback_response():
    """Function to return a fallback response when the model's confidence is below the threshold."""
    return "I'm sorry, I didn't quite understand that. Can you please rephrase?"

# --- Prediction and Confidence Logic ---
def clean_input(sentence):
    """Preprocess the input sentence to convert it into a bag of words."""
    tokens = nltk.word_tokenize(sentence.lower())
    bag = [1 if word in tokens else 0 for word in words]
    return np.array([bag])

def predict_intent(sentence, threshold=0.75):
    """Predict the intent of a given sentence and return the predicted class and confidence."""
    X = clean_input(sentence)
    probs = model.predict(X)[0]
    max_prob = np.max(probs)
    
    if max_prob >= threshold:
        category_index = np.argmax(probs)
        tag = le.inverse_transform([category_index])[0]
        return tag, max_prob
    else:
        return "unknown", max_prob

# --- Final Chatbot Response Function ---
def chatbot_response(input_text):
    """Function to handle user input and return the response based on confidence threshold."""
    tag, confidence = predict_intent(input_text)
    
    if tag == "unknown":
        return get_fallback_response()
    
    # Fetch the intent response from the intents data
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])

# --- Test the Model ---
print("\nTesting model with 75% threshold:")
test_sentences = [
    "Hello there",
    "What time is it?",
    "Thank you for your help"
]

for sentence in test_sentences:
    response = chatbot_response(sentence)
    print(f"Input: '{sentence}' → Response: {response}")

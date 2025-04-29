from flask import Flask, render_template, request, jsonify
from nlp_utils import predict_intent  # Ensure this import is correct

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"]
    
    # Get the prediction and response (including confidence)
    tag, confidence, response = predict_intent(user_msg)
    
    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
# app.py

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer and model from saved_models folder
model_path = os.path.join(os.getcwd(), 'saved_models')

# Check if the directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The directory {model_path} was not found. Please make sure 'saved_models' folder is placed correctly.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create the sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_input = request.form['text']

    # Use the sentiment pipeline to predict the sentiment of the input
    result = sent_pipeline(user_input)

    # Extract the sentiment label (e.g., 'LABEL_0', 'LABEL_1')
    sentiment = result[0]['label']  # 'LABEL_0', 'LABEL_1', or 'LABEL_2'
    
    # Convert sentiment label to a human-readable form (Optional)
    sentiment_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sentiment_label = sentiment_map.get(sentiment, "Unknown")

    return render_template('index.html', sentiment=sentiment_label, user_input=user_input)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    # Get text from the JSON request
    text = request.json.get('text')
    
    # Get the sentiment prediction from the pipeline
    result = sent_pipeline(text)

    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

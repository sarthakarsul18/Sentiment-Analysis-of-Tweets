from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("Models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("Models/vectorizer.pkl", "rb"))

# Initialize the Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the tweet from the form
        tweet = request.form["tweet"]
        
        # Preprocess the tweet using the same preprocessing steps
        tweet_vect = vectorizer.transform([tweet])
        
        # Make a prediction using the model
        prediction = model.predict(tweet_vect)
        
        # Return the result to the user
        sentiment = prediction[0]
        return render_template("index.html", prediction_text=f"Sentiment: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)

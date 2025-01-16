from joblib import load
from spotify_mood_classifier import SpotifyMoodClassifier  # Ensure you import the class

# Load the model
model = load('/Users/shura/FUTURE_ML_01/spotify_mood_classifier/spotify_mood_classifier.joblib')

# If you want to test the model, you can call methods or predictions here
print("Model loaded successfully!")

from spotify_mood_classifier import SpotifyMoodClassifier  # Import your custom class

# Load the model
model = joblib.load('/Users/shura/FUTURE_ML_01/spotify_mood_classifier/spotify_mood_classifier.joblib')

# Your features for prediction (example)
song_features = np.array([0.65, 0.75, -5.0, 0.05, 0.1, 0.0, 0.8, 120]).reshape(1, -1)

# Make the prediction
predicted_mood = model.predict(song_features)

# Print the result
print(f"The predicted mood for the song is: {predicted_mood[0]}")

import pandas as pd
import joblib
from spotify_mood_classifier import SpotifyMoodClassifier  # Import your model class

# Load the trained model
model = joblib.load('/Users/shura/FUTURE_ML_01/spotify_mood_classifier/spotify_mood_classifier.joblib')

# Load the test data
test_data = pd.read_csv('/Users/shura/.cache/kagglehub/datasets/maharshipandya/-spotify-tracks-dataset/versions/1/dataset.csv')

# Selecting the relevant features for prediction
song_features = test_data[['danceability', 'energy', 'acousticness', 'valence', 'tempo']]  # Update this with actual features

# Making predictions using the trained model
predictions = model.predict(song_features)

# Saving predictions to a CSV file
predictions_df = pd.DataFrame({'song_id': test_data['track_id'], 'predicted_mood': predictions})
predictions_df.to_csv('/Users/shura/FUTURE_ML_01/spotify_mood_classifier/data/predictions.csv', index=False)

print("Predictions saved to predictions.csv.")

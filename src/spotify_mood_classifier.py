import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class SpotifyMoodClassifier:
    def __init__(self):
        """Initialize the Spotify Mood Classifier with necessary components."""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = ['danceability', 'energy', 'loudness', 'speechiness',
                              'acousticness', 'instrumentalness', 'valence', 'tempo']

    def preprocess_data(self, data_path):
        """
        Load and preprocess the Spotify dataset.

        Args:
            data_path (str): Path to the dataset directory

        Returns:
            tuple: Processed features and labels
        """
        # Load the dataset from the cached location
        file_path = os.path.join(data_path, 'dataset.csv')
        df = pd.read_csv(file_path)

        # Print initial dataset information
        print("\nInitial Dataset Information:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns in dataset:")
        print(df.columns.tolist())

        # Check for missing values
        missing_values = df[self.feature_columns].isnull().sum()
        print("\nMissing values in features:")
        print(missing_values)

        # Remove duplicates if any
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['track_id'])  # Updated from 'id' to 'track_id'
        print(f"\nRemoved {initial_rows - len(df)} duplicate entries")

        # Handle missing values
        df = df.dropna(subset=self.feature_columns)
        print(f"Removed {initial_rows - len(df)} rows with missing values")

        # Print feature statistics
        print("\nFeature Statistics:")
        print(df[self.feature_columns].describe())

        # Print genre distribution
        print("\nTop 10 genres in the dataset:")
        print(df['track_genre'].value_counts().head(10))

        # Define mood categories based on valence and energy
        df['mood'] = self._categorize_mood(df)

        # Extract features and labels
        X = df[self.feature_columns]
        y = df['mood']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Save preprocessed data info
        self.feature_stats = {
            'total_songs': len(df),
            'mood_distribution': df['mood'].value_counts().to_dict()
        }

        # Save additional metadata for visualization
        self.metadata = {
            'genres': df['track_genre'].unique().tolist(),
            'track_names': df['track_name'].tolist(),
            'artists': df['artists'].tolist()
        }

        return X_scaled, y, df

    def _categorize_mood(self, df):
        """
        Categorize songs into moods based on valence and energy.

        Args:
            df (pd.DataFrame): Input DataFrame with audio features

        Returns:
            pd.Series: Mood labels
        """
        conditions = [
            (df['valence'] >= 0.6) & (df['energy'] >= 0.5),
            (df['valence'] <= 0.4) & (df['energy'] <= 0.5),
            (df['energy'] >= 0.7),
            (df['valence'] >= 0.4) & (df['valence'] <= 0.6)
        ]
        choices = ['happy', 'sad', 'energetic', 'neutral']
        return np.select(conditions, choices, default='neutral')

    def train_model(self, X, y):
        """Train the model with additional logging."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTraining set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")

        self.model.fit(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, X_test, y_test):
        """Evaluate model with additional metrics."""
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)

        # Calculate feature importance
        feature_importance = dict(zip(self.feature_columns,
                                    self.model.feature_importances_))

        print("\nTop important features:")
        for feature, importance in sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.3f}")

        return {
            'classification_report': report,
            'feature_importance': feature_importance
        }

    def visualize_results(self, df, save_dir='visualizations'):
        """Generate enhanced visualizations."""
        os.makedirs(save_dir, exist_ok=True)

        # Mood distribution plot
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='mood')
        plt.title('Distribution of Moods in Spotify Tracks')
        plt.xlabel('Mood Category')
        plt.ylabel('Number of Songs')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mood_distribution.png'))
        plt.close()

        # Feature importance plot
        plt.figure(figsize=(12, 6))
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        sns.barplot(data=importances, x='importance', y='feature')
        plt.title('Feature Importance in Mood Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()

        # Mood vs Energy-Valence scatter plot
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(df['valence'], df['energy'],
                            c=pd.factorize(df['mood'])[0],
                            alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Mood')
        plt.xlabel('Valence')
        plt.ylabel('Energy')
        plt.title('Mood Classification based on Valence and Energy')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mood_scatter.png'))
        plt.close()

def main():
    """Main function to run the Spotify Mood Classifier."""
    print("Starting Spotify Mood Classifier...")

    # Initialize the classifier
    classifier = SpotifyMoodClassifier()

    # Dataset path
    data_path = '/Users/shura/.cache/kagglehub/datasets/maharshipandya/-spotify-tracks-dataset/versions/1'

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, df = classifier.preprocess_data(data_path)

    print(f"\nFinal Dataset Statistics:")
    print(f"Total number of songs: {classifier.feature_stats['total_songs']}")
    print("\nMood distribution:")
    for mood, count in classifier.feature_stats['mood_distribution'].items():
        print(f"{mood}: {count} songs ({count/classifier.feature_stats['total_songs']*100:.1f}%)")

    # Train the model
    print("\nTraining the model...")
    X_train, X_test, y_train, y_test = classifier.train_model(X, y)

    # Evaluate the model
    print("\nEvaluating the model...")
    evaluation = classifier.evaluate_model(X_test, y_test)

    # Generate visualizations
    print("\nGenerating visualizations...")
    classifier.visualize_results(df)
    print("Visualizations saved in 'visualizations' directory")

    # Save the model
    print("\nSaving the model...")
    joblib.dump(classifier, 'spotify_mood_classifier.joblib')
    print("Model saved as 'spotify_mood_classifier.joblib'")

    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()

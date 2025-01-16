import matplotlib.pyplot as plt

# Example mood counts (replace with your actual counts)
mood1_count = 120  # Example count for Mood 1 (Happy)
mood2_count = 80   # Example count for Mood 2 (Sad)
mood3_count = 100  # Example count for Mood 3 (Relaxed)

# List of counts for each mood
mood_distribution = [mood1_count, mood2_count, mood3_count]

# Corresponding mood labels
moods = ['Happy', 'Sad', 'Relaxed']

# Plotting the distribution
plt.bar(moods, mood_distribution, color=['green', 'red', 'blue'])
plt.xlabel('Mood')
plt.ylabel('Count')
plt.title('Distribution of Moods')
plt.show()

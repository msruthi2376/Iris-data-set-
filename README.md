# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic information
print("\nDataset Information:")
print(df.info())

# Summary statistics
print("\nStatistical Summary:")
print(df.describe())

# Count of each species
print("\nNumber of samples per species:")
print(df['species'].value_counts())

# Pairplot to visualize relationships
sns.pairplot(df, hue='species', diag_kind='kde')
plt.suptitle("Iris Dataset - Pairwise Feature Relationships", y=1.02)
plt.show()

# Boxplot for each feature
plt.figure(figsize=(10,6))
sns.boxplot(data=df, orient="h")
plt.title("Distribution of Features in Iris Dataset")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

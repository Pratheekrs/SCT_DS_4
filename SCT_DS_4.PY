import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://example.com/traffic_accidents.csv"  # Replace with actual URL or path
accidents_data = pd.read_csv(url)

# Display the first few rows
print(accidents_data.head())

# Data Cleaning
# Check for missing values
print(accidents_data.isnull().sum())

# Fill missing values (example strategies)
accidents_data['age'].fillna(accidents_data['age'].median(), inplace=True)  # Example for age
accidents_data['weather'].fillna('Unknown', inplace=True)  # Fill missing weather with 'Unknown'
accidents_data.dropna(subset=['time', 'road_condition'], inplace=True)  # Drop rows with crucial missing data

# Convert 'time' to datetime format and extract hour
accidents_data['time'] = pd.to_datetime(accidents_data['time'], format='%H:%M').dt.hour

# Exploratory Data Analysis (EDA)

# 1. Univariate Analysis: Distribution of Road Conditions
plt.figure(figsize=(12, 6))
sns.countplot(y='road_condition', data=accidents_data, order=accidents_data['road_condition'].value_counts().index)
plt.title('Accidents by Road Condition')
plt.xlabel('Count')
plt.ylabel('Road Condition')
plt.show()

# 2. Univariate Analysis: Distribution of Weather Conditions
plt.figure(figsize=(12, 6))
sns.countplot(y='weather', data=accidents_data, order=accidents_data['weather'].value_counts().index)
plt.title('Accidents by Weather Condition')
plt.xlabel('Count')
plt.ylabel('Weather Condition')
plt.show()

# 3. Bivariate Analysis: Accidents by Time of Day
plt.figure(figsize=(12, 6))
sns.countplot(x='time', data=accidents_data)
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(range(24))
plt.show()

# 4. Hotspot Visualization: Accident Locations
plt.figure(figsize=(12, 8))
plt.hexbin(accidents_data['longitude'], accidents_data['latitude'], gridsize=30, cmap='Reds', mincnt=1)
plt.colorbar(label='Number of Accidents')
plt.title('Accident Hotspots')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 5. Correlation Analysis: Numeric Features
plt.figure(figsize=(10, 6))
sns.heatmap(accidents_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Conclusion
print("Analysis complete. The visualizations reveal the relationships between accidents and factors such as road conditions, weather, and time of day.")

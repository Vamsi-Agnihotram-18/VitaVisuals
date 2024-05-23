import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Step 1: Data Loading
data = pd.read_csv('patient_data.csv')

# Step 2: Data Cleaning
data.drop_duplicates(inplace=True)  # Remove duplicates
data.dropna(inplace=True)  # Handling missing values by deletion
data['Symptoms'] = data['Symptoms'].str.lower().str.strip()  # Normalize text data

# Step 3: Data Transformation
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'])  # Convert to datetime
data['Age'] = data['DateOfBirth'].apply(lambda x: (datetime.now() - x).days // 365)  # Calculate age

# Step 4: Feature Engineering
data['Has_Fever'] = data['Symptoms'].apply(lambda x: 1 if 'fever' in x else 0)  # Create binary indicator for fever

# Step 5: Descriptive Statistics
print(data.describe())  # Summary statistics

# Step 6: Correlation Analysis
print(data.corr())  # Correlation matrix

# Step 7: Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Has_Fever', y='Age', data=data)
plt.title('Age Distribution by Fever Presence')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Diagnosis', data=data)
plt.title('Age vs Diagnosis')
plt.show()

# Step 8: Trend Analysis
data['Year'] = data['DateOfBirth'].dt.year
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Age', data=data.groupby('Year')['Age'].mean().reset_index())
plt.title('Average Age Trend over Years')
plt.show()

# Step 9: Interactive Dashboard (conceptual, not executable in standard Python script)
# This would typically be implemented in a web framework or a specialized library like Dash or Plotly.

# Step 10: Insights and Reporting
# Assuming this would be in a real report or interactive dashboard:
print("Insights and actionable reports would be generated here based on the EDA.")


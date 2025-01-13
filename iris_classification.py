import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the Iris dataset from a CSV file
data = pd.read_csv('Iris.csv')
df = pd.read_csv('Iris.csv')

# Display the first 5 rows of the dataset
print(data.head())

print(data.info())
print(data.shape)

# Check for missing values in each column
print(data.isnull().sum())

print(data.describe())

print(data['Species'].value_counts())

# Exploratory Data Analysis
# Histograms
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[feature], kde=True, color='cornflowerblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Boxplots
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='Species', y=feature, palette='Set2')
    plt.title(f'Boxplot of {feature} by Species')
    plt.xlabel('Species')
    plt.ylabel(feature)
    plt.show()

# Scatter Plots
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='SepalLengthCm', y='PetalLengthCm', hue='Species', palette='Set1')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Bar Chart: Count of Species
species_counts = df['Species'].value_counts()
plt.figure(figsize=(6, 4))
species_counts.plot(kind='bar', color=['slategrey', 'lightsteelblue', 'cornflowerblue'])
plt.title('Count of Each Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Pie Chart: Proportions of Species
plt.figure(figsize=(6, 6))
species_counts.plot(kind='pie', autopct='%1.1f%%', colors=['slategrey', 'lightsteelblue', 'cornflowerblue'], startangle=90)
plt.title('Proportion of Each Species')
plt.ylabel('')  # Remove y-axis label for better visualization
plt.show()

# Data Preprocessing
# Dropping the unuseful features and setting target variable
X = df.drop(columns=['Id', 'Species'], axis=1)
Y = df['Species']
print(X.shape)
print(X.head())
print(Y.head())
print(X.info())

# Drop the 'Id' column if it exists in the dataset
if 'Id' in data.columns:
    data = data.drop(columns=['Id'])

# Encode the target variable ('Species') using LabelEncoder
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Splitting features and target variable
X = data.drop(columns=['Species'])  # Features (input features)
y = data['Species']  # Target variable (output)

# Splitting the data into training and testing sets with a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Building
# Using Random Forest Classifier with 100 estimators
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\nClassification Report:")  # Display classification metrics
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")  # Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nAccuracy Score:")  # Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy * 100:.2f}%")

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', color='cornflowerblue')
plt.title("Feature Importance")  # Display bar plot for feature importance
plt.ylabel("Importance Score")
plt.show()


# Save the trained model as a .pkl file
import joblib
joblib.dump(model, 'iris_model.pkl')

print("\nModel saved as 'iris_model.pkl'")  # Confirmation message



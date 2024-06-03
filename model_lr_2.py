import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import joblib  # Use joblib to save the model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression

# Read the Excel file into a DataFrame
data = pd.read_excel(r"G:\\College\\Graduation Project\\model_deployment\\stress_detection\\stress.xlsx")
# data = pd.read_csv("stress.csv")

print(data.shape)

# Rename columns for better readability
data.rename(columns={'rr': 'respiration rate', 't': 'body temperature',
                     'bo': 'blood oxygen', 'sh': 'sleeping hours',
                     'hr': 'heart rate', 'sl': 'stress level'}, inplace=True)

# Extract features and target, only keep 'blood oxygen', 'sleeping hours', 'heart rate'
X = data[['blood oxygen', 'sleeping hours', 'heart rate']].astype(int)  # Convert to int
y = data['stress level']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame with the scaled features
newdf = pd.DataFrame(X_scaled, columns=['blood oxygen', 'sleeping hours', 'heart rate'])
newdf['stress level'] = y

# Check the correlation with the stress level
correlations = newdf.corrwith(newdf['stress level'], method='pearson')
print(correlations)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = tts(newdf.drop(columns=['stress level']), newdf['stress level'], test_size=0.4, random_state=42)

# Train Logistic Regression model
lrr = LogisticRegression()
lrr.fit(x_train, y_train)

# Evaluate the model
accuracy = lrr.score(x_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Save the model to a .pkl file
joblib.dump(lrr, 'stress_detection_model2.pkl')
joblib.dump(scaler, 'scaler2.pkl')

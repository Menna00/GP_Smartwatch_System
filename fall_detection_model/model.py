import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Read the Excel file into a DataFrame
df_train = pd.read_excel(r'acc_gyr.xlsx')

df_train.head()   #first 5 rows of the data

df_train.shape

df_train.isnull().sum()

#Now Visualize the class Distribution

plt.figure(figsize=(12,6))
axis=sns.countplot(x="label",data=df_train)
plt.xticks(x=df_train['label'],rotation='vertical')
plt.show()

# Define the column and the string values to look for
column_to_delete_from = 'label'
values_to_match = ('fall', 'light', 'sit', 'walk')

# Define the proportion of rows to delete
proportion_to_delete = 0.7  # 70% of the matching rows

# Filter rows where the column matches any of the string values
matching_indices = df_train[df_train[column_to_delete_from].isin(values_to_match)].index

# Debugging: print the matching indices
print("\nMatching indices where column '{}' equals '{}':".format(column_to_delete_from, values_to_match))
print(matching_indices)

# Calculate the number of rows to delete
num_rows_to_delete = int(np.ceil(proportion_to_delete * len(matching_indices)))

# Debugging: print the number of rows to delete
print("\nNumber of rows to delete:", num_rows_to_delete)

# Check if there are rows to delete
if num_rows_to_delete > 0:
    # Randomly select indices from the matching indices
    indices_to_delete = np.random.choice(matching_indices, size=num_rows_to_delete, replace=False)

    # Debugging: print the indices to delete
    print("\nIndices to delete:")
    print(indices_to_delete)

    # Drop the selected rows
    df_train = df_train.drop(indices_to_delete)

# Print the modified DataFrame
print("\nDataFrame after randomly deleting rows where column '{}' equals '{}':".format(column_to_delete_from, values_to_match))
print(df_train)


#Now Visualize the class Distribution

plt.figure(figsize=(12,6))
axis=sns.countplot(x="label",data=df_train)
plt.xticks(x=df_train['label'],rotation='vertical')
plt.show()

df_train.describe().T

# Encode the labels
label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['label'])

# Scale the features
scaler = StandardScaler()
features = df_train[['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']]
features_scaled = scaler.fit_transform(features)

print(df_train['label'].unique())

# Prepare the sequences
def create_sequences(features, labels, seq_length=10):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x_seq = features[i:i + seq_length]
        y_seq = labels.iloc[i + seq_length]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(features_scaled, df_train['label'], seq_length)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(label_encoder.classes_))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, 6), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('fall_detection_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load the model
model = tf.keras.models.load_model('fall_detection_model.h5')

# Load the dataset for scaling and label encoding
data = pd.read_excel(r'acc_gyr.xlsx')

# Initialize the scaler and label encoder
scaler = StandardScaler()
scaler.fit(data[['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']])
label_encoder = LabelEncoder()
label_encoder.fit(data['label'])

def predict_sensor_data(readings):
    # Convert readings to a numpy array and scale
    readings_scaled = scaler.transform(np.array(readings).reshape(1, -1))

    # Create a sequence of length 10
    readings_seq = np.repeat(readings_scaled, 10, axis=0).reshape(1, 10, 6)

    # Predict using the model
    prediction = model.predict(readings_seq)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]

# Example sensor readings
example_readings = [7.04, -2.3, -6.28, -11.6, 6.65, 16.48]

# Get the prediction
predicted_label = predict_sensor_data(example_readings)
print(f"Predicted Label: {predicted_label}")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


# Useful Constants
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

DATA_PATH = "data/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
TRAIN = "train/"
TEST = "test/"

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        with open(signal_type_path, 'r') as file:
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
    return y_ - 1

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

training_data_count = len(X_train)
test_data_count = len(X_test)
n_steps = len(X_train[0])
n_input = len(X_train[0][0])

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(LABELS))
y_test = to_categorical(y_test, num_classes=len(LABELS))

# Build LSTM Model
model = Sequential()
model.add(LSTM(32, input_shape=(n_steps, n_input), return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(len(LABELS), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0025), loss='categorical_crossentropy', metrics=['accuracy'])

# Model checkpoint callback
checkpoint_filepath = 'best_model.weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=1500, validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback])

# Load the best weights
model.load_weights(checkpoint_filepath)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Plotting the training history
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show(block=False)
plt.savefig('training_history.png')

# Predict and evaluate
print("Starting prediction...")
y_pred = model.predict(X_test, batch_size=32)
print("Prediction completed.")

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Calculating metrics...")
precision = metrics.precision_score(y_true_classes, y_pred_classes, average='weighted') * 100
recall = metrics.recall_score(y_true_classes, y_pred_classes, average='weighted') * 100
f1_score = metrics.f1_score(y_true_classes, y_pred_classes, average='weighted') * 100

print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1_score:.2f}%")

# Confusion Matrix
print("Calculating confusion matrix...")
confusion_matrix = metrics.confusion_matrix(y_true_classes, y_pred_classes)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
print("Confusion matrix calculated.")

plt.figure(figsize=(12, 8))
plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.rainbow)
plt.title('Confusion matrix (normalized to % of total test data)')
plt.colorbar()
tick_marks = np.arange(len(LABELS))
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show(block=False)
plt.savefig('confusion_matrix.png')

# Save the model
model.save('fall_detection_model.h5')

# Code to load the model and use it for prediction
def load_and_predict(input_data):
    model = tf.keras.models.load_model('fall_detection_model.h5')
    predictions = model.predict(input_data)
    return predictions

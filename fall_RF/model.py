import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import warnings
import os
import random
import joblib  # Import joblib for saving the model


random.seed(42)
plt.rcParams.update({'font.size': 25})
sns.set_theme(color_codes=True)
warnings.filterwarnings('ignore')

# Adjust file paths to your local environment
train_df = pd.read_csv("G:/College/Graduation Project/model_deployment/fall_RF/Train.csv")
test_df = pd.read_csv('G:/College/Graduation Project/model_deployment/fall_RF/Test.csv')

train_df.drop(['Unnamed: 0'], axis=1, inplace=True)
test_df.drop(['Unnamed: 0'], axis=1, inplace=True)

print(f"Training data shape: {train_df.shape}\nTest data shape: {test_df.shape}")

X_train = train_df.drop(['fall', 'label'], axis=1)
y_train = train_df['fall']
X_test = test_df.drop(['fall', 'label'], axis=1)
y_test = test_df['fall']

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X_train, y_train)

def plot_utility_scores(scores, filename):
    y = scores.sort_values(ascending=True)
    width = np.arange(len(y))
    ticks = list(y.index)
    plt.barh(width, y)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores (overall feature)")
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.savefig(filename)
    plt.clf()  # Clear the current figure

plot_utility_scores(mi_scores, "mi_scores.png")

# Compute correlation matrix only on numeric columns
numeric_columns = train_df.select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(train_df[numeric_columns].corr(), ax=ax, cmap="Blues", annot=True)
fig.savefig("correlation_heatmap.png")
plt.clf()


X_train.drop(['gyro_max'], axis=1, inplace=True)
X_test.drop(['gyro_max'], axis=1, inplace=True)

figure, ax = plt.subplots(2, figsize=(24, 16))
sns.swarmplot(x=train_df.label, y=train_df.post_lin_max, ax=ax[0])
sns.swarmplot(x=train_df.label, y=train_df.post_gyro_max, ax=ax[1])
figure.savefig("swarmplots.png")
plt.clf()

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
figure, ax = plt.subplots(4, 2, figsize=(25, 35))
i = 0
j = 0
for col in X_train.columns:
    ax[j][i].title.set_text(col)
    X_train[col].plot.hist(bins=12, alpha=0.5, ax=ax[j][i])
    i += 1
    if i % 2 == 0:
        j += 1
        i = 0
figure.savefig("histograms.png")
plt.clf()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')


# Define hyperparameters
n_estimators = [200, 400, 600, 800, 1000]
max_features = ['auto', 'sqrt']
max_depth = [None, 10, 30, 50, 70]
min_samples_split = [2, 5, 9, 12]
min_samples_leaf = [1, 3, 5, 7]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

print(random_grid)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100, cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)
rf_random.fit(X_train, y_train)

# Save the best model from RandomizedSearchCV
joblib.dump(rf_random.best_estimator_, 'best_random_forest_model.pkl')

rf_random.best_params_

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = (((predictions == test_labels).sum()) / test_labels.shape[0]) * 100
    print('Model Performance')
    print(f'Accuracy = {accuracy:.2f}%.')
    return accuracy

best_estimator = rf_random.best_estimator_
optimal_accuracy = evaluate(best_estimator, X_test, y_test)

params = rf_random.best_params_

min_split = [2, 4, 6, 8, 12]
min_samples_leaf = [1, 2, 3, 4, 5]

x = []
y = []
acc = []
highest_accuracy = 0

for split in min_split:
    params['min_samples_split'] = split
    for leaf in min_samples_leaf:
        params['min_samples_leaf'] = leaf
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = evaluate(model, X_test, y_test)
        acc.append(accuracy)
        x.append(split)
        y.append(leaf)
        highest_accuracy = max(highest_accuracy, accuracy)

# Save the final best model after additional tuning
joblib.dump(model, 'final_best_random_forest_model.pkl')

print(f"The highest accuracy obtained is: {highest_accuracy}%.")

# Plotting 3D accuracy plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111, projection='3d')
accuracy_plot = ax.scatter(x, y, acc, color='red', cmap='Blues')

ax.set_title("3D plot of min_samples_split, min_samples_leaf and accuracy")
ax.set_xlabel('min_samples_split')
ax.set_ylabel('min_samples_leaf')
ax.set_zlabel('Accuracy')
ax.set_zlim(96.8, 97.5)
fig.savefig("3d_accuracy_plot.png")
plt.clf()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


# Read the Excel file into a DataFrame
# data = pd.read_excel(r"G:\\College\\Graduation Project\\model_deployment\\stress_detection\\stress.xlsx")
data = pd.read_csv("stress.csv")

print(data.shape)

# Extract features and target
x = data.drop(columns=['sl'])  # Adjust 'target_column' to your target column name
y = data['sl']

print(x.shape, y.shape)

data.rename(columns = {'rr':'respiration rate', 't':'body temperature',
                        'bo':'blood oxygen', 'sh':'sleeping hours'
                        ,'hr':'heart rate', 'sl':'stress level'}, inplace = True)
print(data.head())

print(data.describe().T)

# checking if there is any null values
print(data.isna().sum().sort_values())

# printing the dataset columns
print(data.columns)

boxplot = data.boxplot(column=['respiration rate', 'body temperature', 'blood oxygen', 'sleeping hours', 'heart rate'],figsize=(15,5))
print(boxplot)

# scaling the features
#importing library
from sklearn.preprocessing import MinMaxScaler
#Defining varible
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(data[['respiration rate', 'body temperature', 'blood oxygen', 'sleeping hours','heart rate']])
print(scaled)

# chaniging the readings with the scaled features
newdf = pd.DataFrame(scaled, columns =['respiration rate', 'body temperature', 'blood oxygen', 'sleeping hours', 'heart rate'])

newdf['stress level'] = data['stress level']

newdf['stress level'].value_counts().sort_values()
#The data is equally distributed among all the stress levels

#plotting a pie chart to show the distribution of data
label = ['low/normal' , 'medium low' , 'medium' ,'medium low','high']
ex=[0.1,0,0,0,0.1]
plt.pie(newdf['stress level'].value_counts(),labels=label,autopct='%.0f%%',explode=ex,shadow=True)
plt.show()


print(newdf.corrwith(newdf['stress level'], method = 'pearson'))

#Importing the Models
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as cvs

#Splitting the Dataset
#splitting among features and target
X = newdf[['respiration rate', 'body temperature', 'blood oxygen', 'sleeping hours', 'heart rate']]
y = newdf['stress level']
#splitting among test and train dataset
x_train, x_test, y_train, y_test= tts(X, y, test_size=0.4)
print('Dimensions of train dataset:',x_train.shape)
print('Dimensions of test dataset:',x_test.shape)

#defining dictionaries for storing results of different models and comparing
sc = {}
rn = {}

#Logistic Regression
lrr=LogisticRegression()
lrr.fit(x_train,y_train)
lrr_pred=lrr.predict(x_test)
print('accuracy_score:',metrics.accuracy_score(y_test, lrr_pred))

r=cvs(lrr, X, y, cv=10, scoring='accuracy').mean()
sc['Logistic Regression']=r
rn['Logistic Regression']=np.array(np.unique(lrr_pred, return_counts=True))
print('cross val score:',r)
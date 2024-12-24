# Here we are predicting diabetes for females only 
# here 0 represents non-diabetic and 1 represents diabetics
# importing dependencies or modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data collection and analysis using a diabetes.csv file taken from kaggle 

# loading the diabetes dataset to a pandas Dataframe

diabetes_dataset = pd.read_csv('Diabetes Prediction\diabetes.csv')
#  printing the first five rows of the dataset
diabetes_dataset.head()
# print(diabetes_dataset.head())
# getting number of rows and columns present in the dataset.
diabetes_dataset.shape
# print(diabetes_dataset.shape)
# getting the statiscal measures of the data
diabetes_dataset.describe()
# print(diabetes_dataset.describe())
# getting the no.of outcomes that represents diabetic and non-diabetic
diabetes_dataset['Outcome'].value_counts()
# print(diabetes_dataset['Outcome'].value_counts())
# calculating the mean values for the outcomes 0 & 1
diabetes_dataset.groupby('Outcome').mean()
# print(diabetes_dataset.groupby('Outcome').mean())
# separating the data and labels 
X= diabetes_dataset.drop(columns='Outcome',axis=1)
# print(X)
Y= diabetes_dataset['Outcome']
# print(Y)
# data standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
# print(standardized_data)
X=standardized_data
Y=diabetes_dataset['Outcome']
# print(X)
# print(Y)


# Now we have to split our data into training and testing data
X_train ,X_test, Y_train , Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# Training the model
classifier = svm.SVC(kernel='linear')


# training the svm classifier
classifier.fit(X_train,Y_train)


# Model Evaluation

# accuracy score on the training data

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction ,Y_train )
print("Accuracy score of the training data :" , training_data_accuracy)


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction , Y_test )

print("Accuracy score of the test data : " ,testing_data_accuracy)


# Making a predictive system that predicts whether an person has diabetics or not 


input_data = (4,110,92,0,0,37.6,0.191,30)

# changing input data to numpy array 


input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#Standardized the input data

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)


if (prediction [0] ==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

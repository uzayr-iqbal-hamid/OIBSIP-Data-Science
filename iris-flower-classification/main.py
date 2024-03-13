import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#if you are running this on jupyter notebook or google colab include %matplotlib inline

# columns
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']


# loading the data from iris.data
df = pd.read_csv('iris.data', names=columns)


#to see 5 rows from the table
df.head()


# statistical analysis
df.describe()


# visualizing the data set
sns.pairplot(df, hue='Class_labels')


# Seperate features and target  
features = df.values
X = features[:,0:4]
Y = features[:,4]


# Average of each feature
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# Plotting the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# Splitting the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# SVM algorithm - Support Vector Machine algorithm
from sklearn.svm import SVC
support_vector = SVC()
support_vector.fit(X_train, y_train)


# Predict from the test dataset
prediction_test = support_vector.predict(X_test)


# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction_test)


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction_test))

final = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction_inputV = support_vector.predict(final)
print("Prediction of Species: {}".format(prediction_inputV))


# Save the model
import pickle
with open('SVM.pickle', 'wb') as file:
    pickle.dump(support_vector, file)


# Load the model
with open('SVM.pickle', 'rb') as file:
    model = pickle.load(file)

model.predict(final)

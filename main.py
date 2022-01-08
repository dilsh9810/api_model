# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#Load the dataset
df = pd.read_csv("Gardening Crops.csv")


#Get first 5 rows of the dataset
df.head()
df.info()


x = df.iloc[:,:].values
z = pd.DataFrame(x)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Convert categorical data into numerical data

labelencoder_x = LabelEncoder()

x[:,1] = labelencoder_x.fit_transform(x[:,1])

z = pd.DataFrame(x)


labelencoder_x = LabelEncoder()

x[:,3] = labelencoder_x.fit_transform(x[:,3])

z = pd.DataFrame(x)


labelencoder_x = LabelEncoder()

x[:,4] = labelencoder_x.fit_transform(x[:,4])

z = pd.DataFrame(x)


#split the modified dataset into the target variable
x = z.iloc[:,0:6]
y = z.iloc[:,-1]

#x["Temperature"] = x.Temperature.astype(float)
#x["PH"] = x.PH.astype(float)
#x["Space"] = x.Space(float)

print(x.dtypes)
print(y.dtypes)
print(z.dtypes)
#print(x)

#Build a naive bayes classifier model
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 1)

#Display train and test values

print(x_train)
print(y_test)
print(x_test)

#print(x_train.dtypes)
#print(x_test.dtypes)
#print(df.dtypes)


#Train the model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)

Predict = gnb.predict(x_test)

print(Predict)

accuracy = accuracy_score(y_test,Predict)

pickle.dump(gnb,open('crop-model.model','wb'))

print('Model Training Finished.\n\tAccuracy obtained: {}'.format(accuracy * 100))














import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fraudData = pd.read_csv("C:/My Files/Excelr/13 - Decision Tree/Assignment/Fraud_check.csv")
fraudData.describe()
fraudData.columns = ['Undergrad', 'MaritalStatus', 'TaxableIncome', 'CityPopulation','WorkExperience', 'Urban']

plt.hist(fraudData['TaxableIncome'])
plt.boxplot(fraudData['TaxableIncome'])

fraudData['TaxableIncome'] = np.where(fraudData['TaxableIncome'] <= 30000 , "Risky","Good")
fraudData['TaxableIncome'].value_counts()

from sklearn import preprocessing
prepocess = preprocessing.LabelEncoder()
columns = ["Undergrad","MaritalStatus","Urban"];

for i in columns:
    fraudData[i] = prepocess.fit_transform(fraudData[i])

fraudData.columns
fraudData = fraudData[['Undergrad', 'MaritalStatus', 'CityPopulation','WorkExperience', 'Urban','TaxableIncome']]

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraudData,test_size=0.2)
trainX = train.iloc[:,0:5]
trainY = train.iloc[:,5]
testX = test.iloc[:,0:5]
testY = test.iloc[:,5]

colnames = list(fraudData.columns)
predictors = colnames[:5]
target = colnames[5]

#DT algorithm
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
accuracy_model=np.mean(train.TaxableIncome == model.predict(train[predictors]))
accuracy_model
accuracy=np.mean(preds==test.TaxableIncome)
accuracy
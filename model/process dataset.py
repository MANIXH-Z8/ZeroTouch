import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\GENIUS\OneDrive\Desktop\Samayal daww\data\gesture_data.csv")
print(data)

x=data.drop(columns='label',axis=1)
y=data['label']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=24)

clf=DecisionTreeClassifier()
clf.fit(xtrain,ytrain)

trainingprediction=clf.predict(xtrain)
trainingaccuracy=accuracy_score(ytrain,trainingprediction)

testingprediction=clf.predict(xtest)
testingaccuracy=accuracy_score(ytest,testingprediction)

print(f"Training accuracy : {trainingaccuracy}")
print(f"Testing accuracy : {testingaccuracy}")
data=data.rename(columns={"drag and hold":"hold and drag"})
data['label']=data['label'].replace({"drag and hold":"hold and drag"})
print(data['label'].unique())
print(data.info())
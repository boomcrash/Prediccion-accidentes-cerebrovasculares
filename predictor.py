import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

# binary conversion label

def train_model():
    # Load the data
    df = pd.read_csv('dataset/strokeData.csv')
    # Split the data into train and test
    X = df.drop(['ever_married','gender','work_type','Residence_type','smoking_status','stroke'], axis=1)
    print(X)
    X_procesing=df[['ever_married','gender','work_type','Residence_type','smoking_status']]
    print(X_procesing)

    X_married=pd.get_dummies(X_procesing['ever_married'])
    #print(X_married)
    X_gender=pd.get_dummies(X_procesing['gender'])
    #print(X_gender)
    X_work=pd.get_dummies(X_procesing['work_type'])
    #print(X_work)
    X_residence=pd.get_dummies(X_procesing['Residence_type'])
    #print(X_residence)
    X_smoking=pd.get_dummies(X_procesing['smoking_status'])
    #print(X_smoking)
    
    y = df['stroke']

    X_Final=pd.concat([X,X_married,X_gender,X_work,X_residence,X_smoking,y],axis=1)
    X_Final=X_Final.fillna(method='ffill')
    Y=X_Final['stroke']
    X_Final=X_Final.drop(['stroke'],axis=1)
    X_Final=X_Final.drop(['id'],axis=1)
    print(X_Final)


    X_train, X_test, y_train, y_test = train_test_split(X_Final, Y, test_size=0.3, random_state=0)
    # Train the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = clf.predict(X_test)
    print('y_pred: ', y_pred)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('y_test: ', y_test)
    print('y_train: ', y_train)
    # Save the model
    pickle.dump(clf, open('model.pkl', 'wb'))
   

train_model()


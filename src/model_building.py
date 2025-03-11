import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


data=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/cleaned_rawfile.csv')
#print(data.head())

# split the data into x and y

X=data.drop(columns=['Churn'])
y=data.iloc[:,3:4].values.ravel()
#print(X)
#print(y)

scale=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)


X_train=scale.fit_transform(X_train)
#X_test=scale.transform(X_test)
new_model=RandomForestClassifier(max_depth=10, max_features='sqrt')
new_model.fit(X_train,y_train)

import pickle

# Save model to a file
with open('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/model/telecom_user.pkl', 'wb') as file:
    pickle.dump(new_model, file)
y_test=pd.DataFrame(y_test)
X_test.to_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/x_test.csv',index=False, header=True,)
y_test.to_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/y_test.csv',index=False, header=True,)




from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import json
import os

with open('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/model/telecom_user.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

y_test=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/y_test.csv')
x_test=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/x_test.csv')
scale=StandardScaler()
x_test=scale.fit_transform(x_test)
y_pred=loaded_model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            #'f1_score': f1_score
        }
if not os.path.exists('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/metrics'):
    os.makedirs('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/metrics')
else:
  print("folder already there metrcics one")

with open ('metrics/metric.json','w') as json_file:
    json.dump(metrics_dict,json_file,indent=4)

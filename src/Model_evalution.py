from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error,adjusted_rand_score
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

with open('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/model/telecom_user.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

y_test=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/y_test.csv')
x_test=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/x_test.csv')
scale=StandardScaler()
x_test=scale.fit_transform(x_test)
y_pred=loaded_model.predict(x_test)

print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))
from datasets import load_dataset
import pandas as pd
import os

ds = load_dataset("aai510-group1/telco-customer-churn")
df1=pd.DataFrame(ds['train'])
df2=df=pd.DataFrame(ds['test'])
df3=df=pd.DataFrame(ds['validation'])
finaldf=pd.concat([df1,df2,df3])
finaldf.to_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/rawfile.csv',index=False, header=True,)
print("data_injection process completed")
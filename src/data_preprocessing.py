#dataload
from datasets import load_dataset
import pandas as pd
import os
finaldf=pd.read_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/rawfile.csv')

finaldf.drop('Churn Category',inplace=True,axis=1)

finaldf.reset_index(drop=True)

finaldf.drop('Churn Reason',inplace=True,axis=1)
finaldf.reset_index(drop=True)
finaldf.drop(['Customer ID','Customer Status','Dependents','Zip Code'],inplace=True,axis=1)
finaldf.reset_index(drop=True)
finaldf.drop(['Lat Long','Offer','Payment Method','Quarter','State'],inplace=True,axis=1)
finaldf.reset_index(drop=True)
finaldf.drop(['Country'],inplace=True,axis=1)
finaldf.reset_index(drop=True)
finaldf.drop(['City'],inplace=True,axis=1)
finaldf.reset_index(drop=True)
finaldf['Contract']=finaldf['Contract'].replace({
  'Two Year':'3',
   'One Year':'2',
    'Month-to-Month':'1',


})
finaldf['Gender']=finaldf['Gender'].replace({

'Male':'0',
'Female':'1'

})

finaldf.drop(['Internet Type'],inplace=True,axis=1)

finaldf.reset_index(drop=True)
finaldf['Gender']=finaldf['Gender'].astype('int64')
finaldf['Contract']=finaldf['Gender'].astype('int64')
import seaborn as sns
import matplotlib.pyplot as plt

# Melt the DataFrame
# melted_df = finaldf.melt(var_name='Column_Name', value_name='Value')
# Increase figure size and rotate labels
# plt.figure(figsize=(15, 8))  # Adjust width and height as needed
# sns.boxplot(x='Column_Name', y='Value', data=melted_df)
# plt.xticks(rotation=90, ha='right')  # Rotate 90 degrees

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

import numpy as np
result=np.quantile(finaldf['Population'],[.25,.75])
result

iqr=np.quantile(finaldf['Population'],.75)-np.quantile(finaldf['Population'],.25)
uperfence=iqr-2.5*np.quantile(finaldf['Population'],.25)
lowerfence=iqr+2.5*np.quantile(finaldf['Population'],.75)
uperfence,lowerfence

updated_finaldf=finaldf[(finaldf['Population'] > 30265.0) & (finaldf['Population'] < 87968.5)]

# melted_df = updated_finaldf.melt(var_name='Column_Name', value_name='Value')

# Increase figure size and rotate labels
# plt.figure(figsize=(15, 8))  # Adjust width and height as needed
# sns.boxplot(x='Column_Name', y='Value', data=melted_df)
# plt.xticks(rotation=90, ha='right')  # Rotate 90 degrees

# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

updated_finaldf.to_csv('C:/Users/sgoel/IVP/New folder___/ML_model_training_using_MLOPS_dvc_and_dvclive/data/Cleaned_rawfile.csv',index=False, header=True,)
print("data preprocessing completed")
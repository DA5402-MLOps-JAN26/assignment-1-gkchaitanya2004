import json
import pandas as pd
import yaml
import os
from app import app
from sklearn.metrics import log_loss,accuracy_score


## 1. Load Config File

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


raw_data_path = config['data']['raw_data_file']
prod_dir = config['data']['production_dir']

metadata_path = config['data']['metadata_path']
manifest_path = config['data']['manifest_path']

train_data_path = config['model']['train_data_file']
target_column = config['preprocess']['target_column']

train_data_name = train_data_path.split('/')[-1]
curr_data_ver = train_data_name.split('_')[0]

model_log_path = config['deployment']['model_log_path']
model_path = config['deployment']['model_path']
model_name = model_path.split('/')[-1]
model_version = model_name.split('_')[0]

threshold = config['deployment']['threshold']

results_dir = config['deployment']['results_dir']
results_file = config['deployment']['results_file']


with open(metadata_path, 'r') as f:
    metadata = json.load(f)

columns_renamed = metadata[curr_data_ver]['columns_renamed']
columns_removed = metadata[curr_data_ver]['columns_dropped']
mapping = metadata[curr_data_ver]['mapping']
scaling_method = metadata[curr_data_ver]['scaling_method']
columns_scaled = scaling_method['columns_scaled']
mean = scaling_method['parameters']['mean']
scale = scaling_method['parameters']['scale']

## 2. Clean the Production data (Same as training data)
def clean_prod_data(df):
   
    # df_new = df.drop(columns=columns_removed).copy()
    df_new = df.dropna().reset_index(drop=True)
    for new_col, old_col in columns_renamed.items():
        df_new = df_new.rename(columns={old_col: new_col})
    return df_new


## 3. Scale and Map the Production data 

def scale_and_map_prod_data(df):
    df_new = df.copy()
    df_new['Type'] = df_new['Type'].map(mapping)
    for col in columns_scaled:
        idx = columns_scaled.index(col)
        mean_col = mean[idx]
        scale_col = scale[idx]
        df_new[col] = (df_new[col] - mean_col) / scale_col

    return df_new
    
def predictions(df):

    cols_to_drop = columns_removed 
    if 'Machine failure' in df.columns:
        cols_to_drop.append('Machine failure')

    target_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

    df_new = df.drop(columns=cols_to_drop)

    model = app.model
    pred_proba = model.predict_proba(df_new)
    predictions = (pred_proba[:, 1] >= 0.5).astype(int)
    
    df['Predicted_Type'] = predictions
    df['Predicted_Prob'] = pred_proba[:, 1]

    loss = log_loss(df[target_column], df['Predicted_Prob'])
    accuracy = accuracy_score(df[target_column], df['Predicted_Type'])
    accuracy = round(accuracy*100, 2)
    loss = round(loss, 4)
    df.drop(columns=target_cols, inplace=True, errors='ignore')

    return df, loss, accuracy

def save_results(df,loss,accuracy,status="success"):

    for cols in columns_scaled:
        idx = columns_scaled.index(cols)
        mean_col = mean[idx]
        scale_col = scale[idx]
        df[cols] = df[cols] * scale_col + mean_col

    df = df.rename(columns=columns_renamed)

    for old_val, new_val in mapping.items():
        df['Type'] = df['Type'].replace(new_val, old_val)
    
    df['Predicted_Type'] = df['Predicted_Type'].replace({0: 'No Failure', 1: 'Failure'})

    timestamp = pd.Timestamp.now()
    date = timestamp.date()
    time = timestamp.time()
    save_model_name = model_name.split('_')[1]


    log_data = {
        "Model Version": model_version,
        "Model Algorithm": save_model_name,
        "Deployment Date": date,
        "Deployment Time": time,
        "rows": len(df),
        "log_loss": loss,
        "accuracy": accuracy,
        "threshold": threshold,
        "status": status
    }

    log_df = pd.DataFrame([log_data])
    log_df.to_csv(
        model_log_path, 
        mode='a', 
        header=not os.path.exists(model_log_path), 
        index=False
    )


    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_path = os.path.join(results_dir, results_file)
    df.to_csv(results_path, index=False)

    return df 


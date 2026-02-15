import pandas as pd
import yaml
import json
from sklearn.preprocessing import StandardScaler
import shutil
import os
import hashlib

"""
    My Version Cotrol System : vX.Y.Z
    where X represent raw data version
    where Y represent cleaned data version
    where Z represent final(cleaned + scaled) data version
"""



## 1. Load Config File
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


## 2. Getting Data Directory Paths
data_dir = config['data']['data_dir']
raw_dir = config['data']['raw_dir']
proc_dir = config['data']['processed_dir']
prod_dir = config['data']['production_dir']

raw_data_path = config['data']['raw_data_file']
curr_version = config['data']['curr_version']

metadata_path = config['data']['metadata_path']
manifest_path = config['data']['manifest_path']


## 3 Create a hash for the raw data so that a new version needs to created or not
def get_hash(path):
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

raw_data_hash = get_hash(raw_data_path)

## 4. Update the hash in history file 
history_path = os.path.join(data_dir, ' version_history.json')
is_new_data = False
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        try:
            history = json.load(f)

        except:
            history = {}
else:
    history = {}
    history[raw_data_hash] = {
        'version': curr_version,
        'timestamp': str(pd.Timestamp.now())
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    is_new_data = True

    

if raw_data_hash not in history:
    version_num_split = curr_version.split('v')[1].split('.')
    x_val = int(version_num_split[0])
    x_val = x_val + 1

    new_version = f'v{x_val}.0.0'
    history[raw_data_hash] = {
        'version': new_version,
        'timestamp': str(pd.Timestamp.now())
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    is_new_data = True




## 5. Create Manifest File if it doesn't exist

if is_new_data:
    
    if not os.path.exists(manifest_path):
        with open(manifest_path, 'w') as f:
            f.write(f"Manifest File for {config['project_name']}\n")
            f.write(f"Author: {config['author']}\n\n")
            f.write(f"Version Cotrol System : vX.Y.Z\n")
            f.write(f"\t- where X represent raw data version\n")
            f.write(f"\t- where Y represent cleaned data version\n")
            f.write(f"\t- where Z represent final(cleaned + scaled) data version\n")
            f.write(f"="*80+"\n\n")

    ## 6. Crate a copy of raw data with name change to ver_raw.csv
    new_version = history[raw_data_hash]['version']
    raw_copy_path = os.path.join(raw_dir, f'{new_version}_raw.csv')
    shutil.copy(raw_data_path, raw_copy_path)


    with open(manifest_path, 'a') as f:
        f.write(f"File Name : {new_version}_raw.csv\n")
        f.write(f"\t- Version: {new_version}\n")
        f.write(f"\t- Derived From: {curr_version}\n")
        f.write(f"\t- File Path: {raw_copy_path}\n")
        f.write(f"\t- Created on: {pd.Timestamp.now()}\n")
        f.write(f"\t- Source: {raw_data_path}\n")
        f.write(f"\t- Script Used: src/data_prep.py\n")
        f.write(f"\t- Description : \n ")
        f.write(f"\t\t- Created a copy of the raw data with name change to {new_version}_raw.csv \n")
        f.write(f"\t\t- This is done for to track which source raw file belong to which version and for easier tracking\n\n")

    config['data']['curr_version'] = new_version
    curr_version = new_version
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f,sort_keys=False,default_flow_style=False)

else:
    print(f"Already this raw data exists using that and not logging and creating a new version.")
    raw_copy_path = os.path.join(raw_dir, f'{curr_version}_raw.csv')

## 7. Data Splitting to Train and Production sets
df = pd.read_csv(raw_copy_path)
prod_data_path = os.path.join(prod_dir, 'prod.csv')
if not os.path.exists(prod_data_path):
    prod_df = df[7000:].reset_index(drop=True)
    train_df = df[:7000].reset_index(drop=True)
    prod_df.to_csv(prod_data_path, index=False)

else:
    prod_df = pd.read_csv(prod_data_path)
    train_df = df

## 9. Clean The Train data and increase the Y by 1## 3. Cleaning the Train data
def clean_data(df,version):

    df_new = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    df_new = df_new.dropna().reset_index(drop=True)
    df_new = df_new.rename(columns={'Air temperature [K]': 'air_temp', 'Process temperature [K]': 'process_temp', 'Torque [Nm]': 'torque', 'Tool wear [min]': 'tool_wear',
                                    'Rotational speed [rpm]': 'rotational_speed'})
    
    version_num_split = version.split('v')[1].split('.')
    x_val = int(version_num_split[0])
    y_val = int(version_num_split[1])
    z_val = int(version_num_split[2])
    y_val = y_val + 1

    new_version = f'v{x_val}.{y_val}.{z_val}'
    train_path = os.path.join(proc_dir, f'{new_version}_cleaned.csv')

    
    with open(manifest_path, 'a') as f:
        f.write(f"File Name : {new_version}_cleaned.csv\n")
        f.write(f"\t- File Version: {new_version}\n")
        f.write(f"\t- Derived From: {version}\n")
        f.write(f"\t- File Path: {train_path}\n")
        f.write(f"\t- Created on: {pd.Timestamp.now()}\n")
        f.write(f"\t- Source: {raw_data_path}\n")
        f.write(f"\t- Script Used: src/data_prep.py\n")
        f.write(f"\t- Description : \n ")
        f.write(f"\t\t- Cleaned the train data by removing unnecessary columns \n")
        f.write(f"\t\t- Removed rows with null values \n")
        f.write(f"\t\t- Handled missing values \n")
        f.write(f"\t\t- Renamed some of the columns for better usage.\n\n")
    
    df_new.to_csv(train_path, index=False)
    return df_new,new_version


## 10. Scale and map the Train data
def scale_and_map_data(df,version):
    scaler = StandardScaler()
    num_cols = ['air_temp', 'process_temp', 'torque', 'tool_wear','rotational_speed']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

    version_num_split = version.split('v')[1].split('.')
    x_val = int(version_num_split[0])
    y_val = int(version_num_split[1])
    z_val = int(version_num_split[2])
    z_val = z_val + 1
    new_version = f'v{x_val}.{y_val}.{z_val}'
    train_path = os.path.join(proc_dir, f'{new_version}_scaled.csv')

    columns_renamed = {
        'air_temp': 'Air temperature [K]',
        'process_temp': 'Process temperature [K]',
        'torque': 'Torque [Nm]',
        'tool_wear': 'Tool wear [min]',
        'rotational_speed': 'Rotational speed [rpm]'
    }

    with open(manifest_path, 'a') as f:
        f.write(f"File Name : {new_version}_scaled.csv\n")
        f.write(f"\t- File Version: {new_version}\n")
        f.write(f"\t- Derived From: {version}\n")
        f.write(f"\t- File Path: {train_path}\n")
        f.write(f"\t- Created on: {pd.Timestamp.now()}\n")
        f.write(f"\t- Source: {raw_data_path}\n")
        f.write(f"\t- Script Used: src/data_prep.py\n")
        f.write(f"\t- Description : \n ")
        f.write(f"\t\t- Scaled the numerical columns using StandardScaler \n")
        f.write(f"\t\t- Mapped the categorical column 'Type' to numerical values.\n\n")

    metadata = {
        "columns": df.columns.tolist(),
        "columns_renamed": columns_renamed,
        "columns_dropped": ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
        "mapping": {'L': 0, 'M': 1, 'H': 2},
        "scaling_method": {
            "method": "StandardScaler",
            "columns_scaled": num_cols,
            "parameters": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist()
            }
        }
    }    

    if os.path.exists(metadata_path):

        with open(metadata_path, 'r') as f:
            try:
                existing_metadata = json.load(f)
            except:
                existing_metadata = {}
    else:
        existing_metadata = {}

    existing_metadata[new_version] = metadata
    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f, indent=4)
        
    df.to_csv(train_path, index=False)
    return df,new_version
        

## 11. Save the final train data
if is_new_data:
    cleaned_train_df,version_clean = clean_data(train_df,curr_version)
    final_train_df,version_final = scale_and_map_data(cleaned_train_df,version_clean)




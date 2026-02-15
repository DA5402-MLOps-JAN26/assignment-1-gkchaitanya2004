import pandas as pd
import yaml
import random
import os
import requests
import time
import json

random.seed(42)


## 1. Load Config File
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


## 2. Loading Raw Data
raw_dir = config['data']['raw_dir']
raw_data_file = config['data']['raw_data_file']
prod_dir = config['data']['production_dir']
df = pd.read_csv(raw_data_file)

## 3. Load Model Metadata and version
model_deploy_log_path = config['deployment']['model_log_path']
deploy_df = pd.read_csv(model_deploy_log_path)
model_name= config['deployment']['model_path'].split('/')[-1]
model_version = model_name.split('_')[0]


## 4. selecting 1500 rows at random for monitoring
row_idxs = random.sample(range(len(df)), 4000)
monitor_df = df.iloc[row_idxs].reset_index(drop=True)

## 5. modify values in the rows to create some drift
day2_part_1 = monitor_df[:1000].copy()
day2_part_2 = monitor_df[1000:2000].copy()
day2_part_3 = monitor_df[2000:3000].copy()
day2_part_4 = monitor_df[3000:].copy()


day2_part_1['Air temperature [K]'] = day2_part_1['Air temperature [K]'] + 12
day2_part_1['Process temperature [K]'] = day2_part_1['Process temperature [K]'] + 12

day2_part_2['Torque [Nm]'] = day2_part_2['Torque [Nm]'] + 15
day2_part_2['Tool wear [min]'] = day2_part_2['Tool wear [min]'] + 4


day2_part_3['Air temperature [K]'] = day2_part_3['Air temperature [K]'] - 5
day2_part_3['Process temperature [K]'] = day2_part_3['Process temperature [K]'] - 7


day2_part_4['Torque [Nm]'] = day2_part_4['Torque [Nm]'] + 10
day2_part_4['Tool wear [min]'] = day2_part_4['Tool wear [min]'] - 2


day2_part_1.to_csv(os.path.join(prod_dir, 'day2_part_1.csv'), index=False)
day2_part_2.to_csv(os.path.join(prod_dir, 'day2_part_2.csv'), index=False)
day2_part_3.to_csv(os.path.join(prod_dir, 'day2_part_3.csv'), index=False)
day2_part_4.to_csv(os.path.join(prod_dir, 'day2_part_4.csv'), index=False)


## 6. Getting model performance metrics from the metadata file for monitoring
model_data = deploy_df[(deploy_df['Model Version'] == model_version) & (deploy_df['status'] == 'success')].iloc[-1]
model_accuracy = model_data['accuracy']
model_log_loss = model_data['log_loss']
model_threshold = model_data['threshold']

## 7. Production Error Rate
def per(loss_actual,loss_pred):
    return (loss_pred - loss_actual)/loss_actual


## 7. Perodic hitting of API with 4 parts of dat for a min with a interval of 15 seconds
drifted_data_files = []
acc_threshold = 10
loss_threshold = 0.5
start_time = time.time()

print("Monitoring Started...")
while time.time() - start_time < 40:
    for i in range(1, 5):
        file_name = f'day2_part_{i}.csv'
        file_path = os.path.join(prod_dir, file_name)
        url = f"http://127.0.0.1:5000/results"
        files = {'files': open(file_path, 'rb')}
        response = requests.post(url, files=files)
        deploy_df = pd.read_csv(model_deploy_log_path)
        if response.status_code == 200:
            last_row = deploy_df.iloc[-1]
            log_loss = last_row['log_loss']
            accuracy = last_row['accuracy']
            per_log_loss = per(model_log_loss, log_loss)
            acc_drift = model_accuracy-accuracy 
            
            print(f"{i}/4 files processed")
            if acc_drift > acc_threshold or per_log_loss > loss_threshold:
                drifted_data_files.append(file_name)
                print(f"Drift Detected in {i}th file: Accuracy Drift = {acc_drift:.2f}%, PER = {per_log_loss:.2f}")
            
            time.sleep(10)

print("Monitoring Ended.")

if len(drifted_data_files) == 0:
    print("No Drift Detected in any of the files.")
else:
    print("Drift Detected in the following files. Recommend retraining on these files:")
    for file in drifted_data_files:
        print(file)
        temp_df = pd.read_csv(os.path.join(prod_dir, file))
        df = pd.concat([df, temp_df], ignore_index=True)
    
df.to_csv(os.path.join(raw_dir, 'new_data.csv'), index=False)
        

       

            

            




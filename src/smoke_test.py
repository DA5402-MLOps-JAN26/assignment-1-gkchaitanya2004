import requests
import yaml
import pandas as pd
import os


BASE_URL = "http://127.0.0.1:5000"

def api_call(file):
    url = f"{BASE_URL}/results"
    files = {'files': open(file, 'rb')}
    response = requests.post(url, files=files)
 
    return response


## 1. Incorrect data format 

org_file = "data/production/prod.csv"
df = pd.read_csv(org_file)

df_new = df.drop(columns=['Type'])
os.makedirs("data/smoke_tests", exist_ok=True)

file_path = "data/smoke_tests/prod.csv"
df_new.to_csv(file_path, index=False)

response = api_call(file_path)
if response.status_code == 400:
    print("Test Passed: The API correctly returned an error for incorrect data structure in a .csv file.")



## 2. Incorrect file type

file_path = "data/smoke_tests/prod.txt"
with open(file_path, 'w') as f:
    f.write("This is just a test file.Model should return an error for this file type")

response = api_call(file_path)
if response.status_code == 400:
    print("Test Passed: The API correctly returned an error for incorrect file type.")


## 3. Empty file

empty_file_path = "data/smoke_tests/empty_prod.csv"
pd.DataFrame().to_csv(empty_file_path, index=False)

response = api_call(empty_file_path)
if response.status_code == 400:
    print("Test Passed: The API correctly returned an error for empty file.")



## 4. Valid file

response = api_call(org_file)
if response.status_code == 200:
    print("Test Passed: The API correctly processed the valid file.")



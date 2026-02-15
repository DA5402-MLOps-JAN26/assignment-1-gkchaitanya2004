from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,log_loss
from sklearn.model_selection import train_test_split
import subprocess
import pandas as pd
import yaml
import os
import json
import joblib


## 1. Generate git hash
def get_git_hash():
    try:
        hash = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True)
        return hash.stdout.strip()
    except Exception as e:
        return "unknown"

## 1. Config and Data Loading
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_data_path = config['model']['train_data_file']
df = pd.read_csv(train_data_path)


## 2. Data Splitting
target_column = config['preprocess']['target_column']
metadata_path = config['model']['metadata_path']
curr_data_ver = train_data_path.split('/')[-1]

## 3. Additional parameters for model training
params = config['model_params']
class_weights = params['class_weights']
val_size = params['val_size']
random_state = params['random_state']
max_iter = params.get('max_iter', 1000)
max_depth = params.get('max_depth', None)
stratify_col = config['preprocess']['stratify']
model_algorithm = params['algorithm']


X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=df[stratify_col])


## 4. Model Training
match_found = False
model_ver = None


if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        try:
            history = json.load(f)
        except:
            print("Metadata file is empty or corrupted. Starting with an empty history.")
            history = {}

        for version, entry in history.items():
            entry_type = entry['model_algorithm']
            entry_data_ver = entry['data_version']
            entry_model_param = entry['model_params']

            if entry_type == model_algorithm and entry_data_ver == curr_data_ver and entry_model_param == params:
                match_found = True
                model_ver = version
                break

else:
    history = {}


if match_found:
    print("Model already has trained. Loading the saved model.")
    model_path = os.path.join(config['model']['model_dir'], f'{model_ver}_{model_algorithm}_model.joblib')
    model = joblib.load(model_path)

else:
    model = LogisticRegression(class_weight=class_weights, max_iter=max_iter, random_state=random_state)
    # model = RandomForestClassifier(class_weight=class_weights, max_depth=10, random_state=random_state)
    model.fit(X_train, y_train)

## 5. Model Evaluation
y_pred = model.predict(X_val)
report = classification_report(y_val, y_pred, output_dict=True)
loss = log_loss(y_val, model.predict_proba(X_val))
accuracy = report['accuracy']
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']


## 6. Metadata Logging (Only if any change is made to the model)
model_dir = config['model']['model_dir']

if not match_found:

    if history:

        model_curr_ver = max(history.keys(),key = lambda x: int(x.split('v')[1]))
        model_num = model_curr_ver.split('v')[1]
        model_ver = f"v{int(model_num) + 1}"
    else:
        model_ver = "v1"

    new_entry = {
        'model_algorithm': model_algorithm,
        'Training Date': str(pd.Timestamp.now()),
        'data_version': curr_data_ver,
        'model_params': params,
        'git_hash': get_git_hash(),
        'val_accuracy': round(accuracy*100, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'val_log_loss': round(loss, 4)
    }

    history[model_ver] = new_entry

    with open(metadata_path, 'w') as f:
        json.dump(history, f, indent=4)

    ## 8. Save the model
    model_path = os.path.join(model_dir, f'{model_ver}_{model_algorithm}_model.joblib')
    joblib.dump(model, model_path)

print(f"Model trained and saved as {model_path} with accuracy: {accuracy*100:.2f}%")



                            
```markdown
# MLOps Manual Pipeline Implementation

**Name:** Krishna Chaitanya Gorle
**Roll No:** DA25M011
---

## Project Structure

```text
├── config.yaml # Central configuration for paths and model parameters
├── deployment_logs.csv # Logs generated during inference
├── data/
│ ├── raw/ # Place raw training data here
| ├── processed/ # Preprocesed DAta
│ └── production/ # Place for production data here
└── src/
 ├── data_prep.py # Scripts for cleaning and preprocessing
 ├── train.py # Logic for training the model
 ├── inference.py # Flask application for model serving
 ├── smoke_test.py # Scripts to verify API health
 └── monitor.py # Drift detection and monitoring script
There are some helper files such as prediction.py which are required for running the model properlu

```

---

## Execution Instructions

Follow the steps below to execute the pipeline in order.

### 1. Data Preparation

This is step where we preprocess data to get the data in required format to train the model 

1. Ensure your raw dataset is placed in the **`data/raw`** directory or the path you defined in `config.yaml`
2. Now run this execution script to preprocess.
```bash
python3 src/data_prep.py

```



### 2. Train the Model

This step trains a Logistic Regression model using parameters defined in `config.yaml`.
1. You can also change parameter and model in `config.yaml` but please make sure you do the relavent import
2. Run the training script:
```bash
python3 src/train.py

```



### 3. Inference (Deployment)

After training the model is ready fotr depyloyment

1. Start the deployment server by running this:
```bash
python3 src/inference.py

```
2. Go to `http://127.0.0.1:<port>` (the port is defined in your `config.yaml`).
3. There you can upload the production data and can see adn download the results
4. **Logs:** After prediction, you can also check the logs of the model in **`deployment_logs.csv`** 
### 4. Smoke Tests

These smoke tests are to ensure the api end points are working properly

1. Execute the test script:
```bash
python3 src/smoke_test.py

```

### 5. Monitoring & Retraining

This step checks for data drift and prepares for retraining if necessary.

1. Run the monitoring script:
```bash
python3 src/monitor.py

```
2. If drift is dected it will add a file named `new_data.csv` to your raw_directory and perform the whole process again

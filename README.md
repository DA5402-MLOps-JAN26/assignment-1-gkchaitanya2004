```markdown
# MLOps Manual Pipeline Implementation

**Name:** Krishna Chaitanya Gorle
**Roll No:** DA25M011

This project implements an end-to-end MLOps pipeline to predict wheater a machine was working or not based on the parameters
---

## Project Structure

```text
├── config.yaml              # Central configuration for paths and model parameters
├── deployment_logs.csv      # Logs generated during inference
├── data/
│   ├── raw/                 # Place raw training data here
|   ├──  processed/          # Preprocesed DAta
│   └── production/          # Place for production data here
└── src/
    ├── data_prep.py         # Scripts for cleaning and preprocessing
    ├── train.py             # Logic for training the model
    ├── inference.py         # Flask application for model serving
    ├── smoke_test.py        # Scripts to verify API health
    └── monitor.py           # Drift detection and monitoring script
There are some helper files such as prediction.py which are required for running the model properlu

```

---

## Execution Instructions

Follow the steps below to execute the pipeline in order.

### 1. Data Preparation

This step processes the raw data and prepares it for training.

1. Ensure your raw dataset is placed in the **`data/raw`** directory.
2. Run the data preparation script:
```bash
python3 src/data_prep.py

```



### 2. Train the Model

This step trains a Logistic Regression model using parameters defined in `config.yaml`.

1. (Optional) You can modify model hyperparameters in **`config.yaml`**.
2. (Optional) To use a different algorithm, import and change the model class directly in **`src/train.py`**.
3. Run the training script:
```bash
python3 src/train.py

```



### 3. Inference (Deployment)

This step starts the web server to make predictions on new data.

1. Start the inference server:
```bash
python3 src/inference.py

```


2. Open your browser and navigate to `http://127.0.0.1:<port>` (the port is defined in your `config.yaml`).
3. Upload the production data (located in **`data/prod`**) via the web interface.
4. **Logs:** After prediction, you can verify the logs in **`deployment_logs.csv`** (filename configurable in `config.yaml`).

### 4. Smoke Tests

Run smoke tests to ensure the API and model endpoints are functioning correctly.

1. Execute the test script:
```bash
python3 src/smoke_test.py

```


2. Verify the output in your terminal confirms that all tests have **passed**.

### 5. Monitoring & Retraining

This step checks for data drift and prepares for retraining if necessary.

1. Run the monitoring script:
```bash
python3 src/monitor.py

```


2. If drift is detected, a new training file named **`new_data.csv`** will be generated in the **`data/raw`** directory.
3. **Action Required:** Update the dataset path in **`config.yaml`** to point to this new file.
4. **Redo the process:** Repeat Step 1 (Data Prep) and Step 2 (Train) to retrain the model on the new data.

Now update parameters accordingly in comfig.yaml and redo the process

from flask import Flask
import joblib
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
model_path = config['deployment']['model_path']
model = joblib.load(model_path)
app.model = model

from app import routes


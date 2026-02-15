from app import app
import yaml

if __name__ == "__main__":

    ## 1. Config loading for port and debug mode
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    port = config['deployment']['port']
    debug_mode = config['deployment']['debug']

    ## 2. Running the Flask app
    app.run(port=port, debug=debug_mode)


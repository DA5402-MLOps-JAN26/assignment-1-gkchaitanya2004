from app import app
from flask import render_template,request,send_file
import pandas as pd
from predictions import predictions,clean_prod_data,scale_and_map_prod_data,save_results
import os

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST': 
        try:
            file = request.files['files']
            if file:
                df = pd.read_csv(file)
            
            status = "success"
            df = clean_prod_data(df)
            df = scale_and_map_prod_data(df)
            df, loss, accuracy = predictions(df)
            df = save_results(df,loss,accuracy,status)
            df_display = df.drop(columns=['Predicted_Prob'])
            return render_template('results.html', table=df_display.to_html(index=False)),200
           
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('error.html', error_message=error_message),400

@app.route('/download')
def download():
    path = os.path.join(app.root_path, 'results', 'results.csv')
    return send_file(path, as_attachment=True, mimetype='text/csv', download_name='results.csv')

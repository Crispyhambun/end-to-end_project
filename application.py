from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import Custom_Data, PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = Custom_Data(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert the custom data into a DataFrame
        pred_df = data.get_data_dataframe()
        print(pred_df)  # Debug: Check the data being passed

        # Make predictions using the pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Pass the result to the template as a keyword argument
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")

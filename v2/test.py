from flask import Flask, request, send_file, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if file and allowed_file(file.filename):
        try:
            # Load the data
            df = pd.read_csv(file)
            
            # Load the pre-trained model
            model = load_model('model3.h5')
            
            # Print column names for debugging
            print("Available columns in CSV:", df.columns.tolist())
            
            # Make predictions
            predictions = model.predict(df)
            
            # Create a new dataframe with predictions
            output_df = df.copy()
            output_df['predictions'] = predictions
            
            # encoding the predictions
            output_df['predictions'] = output_df['predictions'].apply(lambda x: 1 if x > 0.5 else 0)
            
            # Get first 20 rows for display
            display_df = output_df.head(20)
            
            # Convert DataFrame to HTML table
            table_html = display_df.to_html(classes='table table-striped table-hover', index=False)
            
            return render_template('results.html', 
                                table=table_html,
                                total_rows=len(output_df),
                                displayed_rows=len(display_df),
                                column_count=len(output_df.columns))
            
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return 'Invalid file format', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

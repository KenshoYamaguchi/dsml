from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import pandas as pd
import joblib
import json
import plotly
from werkzeug.utils import secure_filename

from config import Config
from utils.preprocessing import DataPreprocessor
from utils.train_model import LightGBMTrainer
from utils.evaluate import ModelEvaluator
from utils.predict import PredictionService

app = Flask(__name__)
Config.init_app(app)

# Global variables for session state
current_data = None
preprocessor = None
model = None
feature_columns = None
target_column = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    
    if 'file' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    if file and Config.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load data
            preprocessor_temp = DataPreprocessor()
            current_data = preprocessor_temp.load_data(filepath)
            
            # Store in session
            session['data_file'] = filepath
            
            # Get basic statistics
            stats = preprocessor_temp.get_basic_statistics(current_data)
            
            return render_template('analyze.html', 
                                 columns=current_data.columns.tolist(),
                                 data_preview=current_data.head().to_html(classes='table table-striped'),
                                 stats=stats,
                                 data_shape=current_data.shape)
        except Exception as e:
            flash(f'ファイルの読み込みに失敗しました: {str(e)}')
            return redirect(url_for('index'))
    
    flash('許可されていないファイル形式です')
    return redirect(url_for('index'))

@app.route('/visualize', methods=['POST'])
def visualize_data():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'データが読み込まれていません'})
    
    column = request.json.get('column')
    chart_type = request.json.get('chart_type', 'histogram')
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        if chart_type == 'histogram':
            if current_data[column].dtype in ['object']:
                # Categorical data - value counts
                value_counts = current_data[column].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           labels={'x': column, 'y': '頻度'})
            else:
                # Numerical data - histogram
                fig = px.histogram(current_data, x=column, nbins=30)
            
        elif chart_type == 'box':
            if current_data[column].dtype not in ['object']:
                fig = px.box(current_data, y=column)
            else:
                return jsonify({'error': 'カテゴリカルデータには箱ひげ図は使用できません'})
        
        elif chart_type == 'scatter':
            x_column = request.json.get('x_column')
            y_column = request.json.get('y_column')
            if x_column and y_column:
                fig = px.scatter(current_data, x=x_column, y=y_column)
            else:
                return jsonify({'error': 'X軸とY軸の列を指定してください'})
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'plot': graphJSON})
        
    except Exception as e:
        return jsonify({'error': f'可視化に失敗しました: {str(e)}'})

@app.route('/train', methods=['POST'])
def train_model():
    global current_data, preprocessor, model, feature_columns, target_column
    
    if current_data is None:
        flash('データが読み込まれていません')
        return redirect(url_for('index'))
    
    target_column = request.form.get('target_column')
    feature_columns = request.form.getlist('feature_columns')
    
    if not target_column or not feature_columns:
        flash('目的変数と説明変数を選択してください')
        return redirect(url_for('index'))
    
    try:
        # Preprocessing
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_for_training(current_data, target_column, feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # Train model
        trainer = LightGBMTrainer()
        model = trainer.train_with_hyperparameter_tuning(X_train, y_train, X_test, y_test)
        
        # Save model and preprocessor
        trainer.model = model
        trainer.save_model(Config.MODEL_FILE)
        joblib.dump(preprocessor, Config.PREPROCESSOR_FILE)
        
        # Evaluate model
        evaluator = ModelEvaluator(model, X_test, y_test, feature_names=X.columns.tolist())
        evaluation_report = evaluator.generate_evaluation_report()
        
        # Convert plots to JSON for frontend
        plots_json = {}
        for plot_name, plot_obj in evaluation_report['plots'].items():
            if plot_obj is not None and hasattr(plot_obj, 'to_json'):
                plots_json[plot_name] = json.dumps(plot_obj, cls=plotly.utils.PlotlyJSONEncoder)
            elif isinstance(plot_obj, str):  # Base64 image
                plots_json[plot_name] = plot_obj
        
        return render_template('analyze.html',
                             training_complete=True,
                             metrics=evaluation_report['metrics'],
                             plots=plots_json,
                             feature_importance=evaluator.calculate_feature_importance().to_dict('records'))
        
    except Exception as e:
        flash(f'モデル学習に失敗しました: {str(e)}')
        return redirect(url_for('index'))

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    if file and Config.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load prediction service
            prediction_service = PredictionService(Config.MODEL_FILE, Config.PREPROCESSOR_FILE)
            
            # Generate predictions
            prediction_report = prediction_service.generate_prediction_report(file_path=filepath)
            
            # Convert plots to JSON for frontend
            plots_json = {}
            for plot_name, plot_obj in prediction_report['plots'].items():
                if plot_obj is not None and hasattr(plot_obj, 'to_json'):
                    plots_json[plot_name] = json.dumps(plot_obj, cls=plotly.utils.PlotlyJSONEncoder)
                elif isinstance(plot_obj, str):  # Base64 image
                    plots_json[plot_name] = plot_obj
            
            return render_template('predict.html',
                                 prediction_complete=True,
                                 statistics=prediction_report['statistics'],
                                 plots=plots_json,
                                 predictions_preview=prediction_report['result_dataframe'].head(10).to_html(classes='table table-striped'))
            
        except Exception as e:
            flash(f'予測に失敗しました: {str(e)}')
            return redirect(url_for('predict_page'))
    
    flash('許可されていないファイル形式です')
    return redirect(url_for('predict_page'))

@app.route('/download_predictions')
def download_predictions():
    # This would implement file download functionality
    # For now, just redirect back
    flash('ダウンロード機能は未実装です')
    return redirect(url_for('predict_page'))

@app.errorhandler(413)
def too_large(e):
    flash('ファイルサイズが大きすぎます（最大16MB）')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
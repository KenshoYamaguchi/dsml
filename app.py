from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import pandas as pd
import joblib
import json
import plotly
from werkzeug.utils import secure_filename

from config import Config
from utils.preprocessing import DataPreprocessor
from utils.lgb_train_model import LightGBMTrainer
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

@app.route('/analyze')
def analyze_page():
    global current_data
    if current_data is not None:
        preprocessor_temp = DataPreprocessor()
        stats = preprocessor_temp.get_basic_statistics(current_data)
        return render_template('analyze.html', 
                             columns=current_data.columns.tolist(),
                             data_preview=current_data.head().to_html(classes='table table-striped'),
                             stats=stats,
                             data_shape=current_data.shape)
    return redirect(url_for('index'))

@app.route('/train')
def train_page():
    global current_data
    columns = current_data.columns.tolist() if current_data is not None else None
    return render_template('train.html', columns=columns)

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
            
            return redirect(url_for('analyze_page'))
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
                           labels={'x': column, 'y': '頻度'}, 
                           title=f'{column}の分布')
                fig.update_layout(
                    xaxis_title=column,
                    yaxis_title='頻度',
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            else:
                # Numerical data - histogram
                fig = px.histogram(current_data, x=column, nbins=30, title=f'{column}の分布')
                # Set appropriate range based on data distribution
                q1 = current_data[column].quantile(0.01)
                q99 = current_data[column].quantile(0.99)
                fig.update_layout(
                    xaxis_title=column,
                    yaxis_title='頻度',
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis=dict(range=[q1, q99])
                )
            
        elif chart_type == 'box':
            if current_data[column].dtype not in ['object']:
                fig = px.box(current_data, y=column, title=f'{column}の箱ひげ図')
                # Set appropriate range based on data distribution
                q1 = current_data[column].quantile(0.01)
                q99 = current_data[column].quantile(0.99)
                fig.update_layout(
                    yaxis_title=column,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                    yaxis=dict(range=[q1, q99])
                )
            else:
                return jsonify({'error': 'カテゴリカルデータには箱ひげ図は使用できません'})
        
        elif chart_type == 'scatter':
            x_column = request.json.get('x_column')
            y_column = request.json.get('y_column')
            if x_column and y_column:
                fig = px.scatter(current_data, x=x_column, y=y_column, 
                               title=f'{x_column} vs {y_column}')
                fig.update_layout(
                    xaxis_title=x_column,
                    yaxis_title=y_column,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            else:
                return jsonify({'error': 'X軸とY軸の列を指定してください'})
        
        elif chart_type == 'line':
            if current_data[column].dtype not in ['object']:
                fig = px.line(x=current_data.index, y=current_data[column], 
                             title=f'{column}の時系列プロット')
                fig.update_layout(
                    xaxis_title='インデックス',
                    yaxis_title=column,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            else:
                return jsonify({'error': 'カテゴリカルデータには線グラフは使用できません'})
        
        elif chart_type == 'violin':
            if current_data[column].dtype not in ['object']:
                fig = px.violin(current_data, y=column, title=f'{column}のバイオリンプロット')
                fig.update_layout(
                    yaxis_title=column,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            else:
                return jsonify({'error': 'カテゴリカルデータにはバイオリンプロットは使用できません'})
        
        # Convert to JSON using Plotly's to_json method for better compatibility
        try:
            graphJSON = fig.to_json()
            return jsonify({'plot': graphJSON})
        except Exception as json_error:
            print(f"JSON encoding error: {str(json_error)}")
            # Fallback to old method
            try:
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return jsonify({'plot': graphJSON})
            except Exception as fallback_error:
                print(f"Fallback JSON encoding error: {str(fallback_error)}")
                return jsonify({'error': f'グラフのJSON変換に失敗しました: {str(fallback_error)}'})
        
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
            if plot_obj is not None:
                if hasattr(plot_obj, 'to_json'):
                    try:
                        plots_json[plot_name] = plot_obj.to_json()
                    except Exception as json_error:
                        print(f"Error converting {plot_name} to JSON: {str(json_error)}")
                        try:
                            plots_json[plot_name] = json.dumps(plot_obj, cls=plotly.utils.PlotlyJSONEncoder)
                        except Exception as fallback_error:
                            print(f"Fallback error for {plot_name}: {str(fallback_error)}")
                            plots_json[plot_name] = None
                elif isinstance(plot_obj, str):  # Base64 image
                    plots_json[plot_name] = plot_obj
                else:
                    plots_json[plot_name] = None
        
        return render_template('train.html',
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
                if plot_obj is not None:
                    if hasattr(plot_obj, 'to_json'):
                        try:
                            plots_json[plot_name] = plot_obj.to_json()
                        except Exception as json_error:
                            print(f"Error converting {plot_name} to JSON: {str(json_error)}")
                            try:
                                plots_json[plot_name] = json.dumps(plot_obj, cls=plotly.utils.PlotlyJSONEncoder)
                            except Exception as fallback_error:
                                print(f"Fallback error for {plot_name}: {str(fallback_error)}")
                                plots_json[plot_name] = None
                    elif isinstance(plot_obj, str):  # Base64 image
                        plots_json[plot_name] = plot_obj
                    else:
                        plots_json[plot_name] = None
            
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

@app.route('/download_training_results')
def download_training_results():
    global model, preprocessor, feature_columns, target_column
    
    if model is None:
        flash('学習済みモデルがありません')
        return redirect(url_for('train_page'))
    
    try:
        # Create a summary report
        evaluator = ModelEvaluator(model, None, None, feature_names=feature_columns)
        
        # Generate downloadable content
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Model Training Results'])
        writer.writerow(['Target Variable', target_column])
        writer.writerow(['Feature Variables'] + feature_columns)
        writer.writerow([])
        
        # Would add metrics here
        writer.writerow(['Metrics will be added here'])
        
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=training_results.csv'}
        )
        
    except Exception as e:
        flash(f'ダウンロードに失敗しました: {str(e)}')
        return redirect(url_for('train_page'))

@app.errorhandler(413)
def too_large(e):
    flash('ファイルサイズが大きすぎます（最大16MB）')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Only run in debug mode locally, not in production
    is_production = os.environ.get('WEBSITE_HOSTNAME') is not None
    app.run(debug=not is_production, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
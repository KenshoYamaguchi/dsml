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
    
    # Check if there are saved training results in session
    training_complete = session.get('training_complete', False)
    if training_complete:
        metrics = session.get('training_metrics')
        plot_files = session.get('training_plot_files', {})
        feature_importance = session.get('feature_importance')
        
        # Load plot images from files
        plots = {}
        import base64
        for plot_name, filename in plot_files.items():
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(plot_path):
                with open(plot_path, 'rb') as f:
                    plots[plot_name] = base64.b64encode(f.read()).decode()
        
        return render_template('train.html',
                             columns=columns,
                             training_complete=True,
                             metrics=metrics,
                             plots=plots,
                             feature_importance=feature_importance)
    
    return render_template('train.html', columns=columns)

@app.route('/train/new')
def new_train():
    # Clean up old plot files
    plot_files = session.get('training_plot_files', {})
    for filename in plot_files.values():
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(plot_path):
            os.remove(plot_path)
    
    # Clear existing training results
    session.pop('training_complete', None)
    session.pop('training_metrics', None)
    session.pop('training_plot_files', None)
    session.pop('feature_importance', None)
    session.pop('target_column', None)
    session.pop('feature_columns', None)
    session.pop('training_session_id', None)
    
    return redirect(url_for('train_page'))

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
            
            # Clean up old plot files when new data is uploaded
            plot_files = session.get('training_plot_files', {})
            for filename in plot_files.values():
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(plot_path):
                    os.remove(plot_path)
            
            # Clear any previous training results when new data is uploaded
            session.pop('training_complete', None)
            session.pop('training_metrics', None)
            session.pop('training_plot_files', None)
            session.pop('feature_importance', None)
            session.pop('target_column', None)
            session.pop('feature_columns', None)
            session.pop('training_session_id', None)
            
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import japanize_matplotlib
        import base64
        from io import BytesIO
        
        plt.figure(figsize=(6, 4))
        
        if chart_type == 'histogram':
            if current_data[column].dtype in ['object']:
                # Categorical data - value counts
                value_counts = current_data[column].value_counts().head(20)
                plt.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
                plt.ylabel('頻度')
                plt.title(f'{column}の分布')
            else:
                # Numerical data - histogram
                data_clean = current_data[column].dropna()
                plt.hist(data_clean, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                plt.xlabel(column)
                plt.ylabel('頻度')
                plt.title(f'{column}の分布')
            
        elif chart_type == 'box':
            if current_data[column].dtype not in ['object']:
                data_clean = current_data[column].dropna()
                plt.boxplot(data_clean, patch_artist=True, 
                           boxprops=dict(facecolor='lightblue', alpha=0.7))
                plt.ylabel(column)
                plt.title(f'{column}の箱ひげ図')
                plt.xticks([1], [column])
            else:
                return jsonify({'error': 'カテゴリカルデータには箱ひげ図は使用できません'})
        
        elif chart_type == 'scatter':
            x_column = request.json.get('x_column')
            y_column = request.json.get('y_column')
            if x_column and y_column:
                plt.scatter(current_data[x_column], current_data[y_column], 
                           alpha=0.6, color='steelblue', s=30)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'{x_column} vs {y_column}')
            else:
                return jsonify({'error': 'X軸とY軸の列を指定してください'})
        
        elif chart_type == 'line':
            if current_data[column].dtype not in ['object']:
                plt.plot(current_data.index, current_data[column], 
                        color='steelblue', linewidth=2)
                plt.xlabel('インデックス')
                plt.ylabel(column)
                plt.title(f'{column}の時系列プロット')
            else:
                return jsonify({'error': 'カテゴリカルデータには線グラフは使用できません'})
        
        elif chart_type == 'violin':
            if current_data[column].dtype not in ['object']:
                data_clean = current_data[column].dropna()
                parts = plt.violinplot([data_clean], positions=[1], showmeans=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)
                plt.ylabel(column)
                plt.title(f'{column}のバイオリンプロット')
                plt.xticks([1], [column])
            else:
                return jsonify({'error': 'カテゴリカルデータにはバイオリンプロットは使用できません'})
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64 for web display
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150, 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return jsonify({'plot': image_base64})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        
        # Save plots to files instead of session
        plots_json = evaluation_report['plots']
        feature_importance_data = evaluator.calculate_feature_importance().to_dict('records')
        
        # Save plot images to temporary files
        import uuid
        session_id = session.get('training_session_id', str(uuid.uuid4()))
        session['training_session_id'] = session_id
        
        plot_files = {}
        for plot_name, plot_data in plots_json.items():
            if plot_data:
                plot_filename = f"plot_{session_id}_{plot_name}.png"
                plot_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
                
                # Save base64 image to file
                import base64
                with open(plot_path, 'wb') as f:
                    f.write(base64.b64decode(plot_data))
                plot_files[plot_name] = plot_filename
        
        # Save training results to session (lightweight data only)
        session['training_complete'] = True
        session['training_metrics'] = evaluation_report['metrics']
        session['training_plot_files'] = plot_files
        session['feature_importance'] = feature_importance_data
        session['target_column'] = target_column
        session['feature_columns'] = feature_columns
        
        return render_template('train.html',
                             training_complete=True,
                             metrics=evaluation_report['metrics'],
                             plots=plots_json,
                             feature_importance=feature_importance_data)
        
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
            
            # All plots are now base64 strings from matplotlib
            plots_json = prediction_report['plots']
            
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
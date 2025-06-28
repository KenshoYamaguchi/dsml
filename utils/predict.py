import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import japanize_matplotlib
import base64
from io import BytesIO
from .preprocessing import DataPreprocessor
from .lgb_train_model import LightGBMTrainer
from .evaluate import ModelEvaluator

class PredictionService:
    def __init__(self, model_path=None, preprocessor_path=None):
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
        # Load model and preprocessor if paths provided
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
    
    def load_model(self, model_path):
        trainer = LightGBMTrainer()
        self.model = trainer.load_model(model_path)
        return self.model
    
    def load_preprocessor(self, preprocessor_path):
        import joblib
        self.preprocessor = joblib.load(preprocessor_path)
        return self.preprocessor
    
    def predict_from_file(self, file_path):
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # Load data
        df = self.preprocessor.load_data(file_path)
        
        # Preprocess
        X_processed = self.preprocessor.preprocess_for_prediction(df)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['predictions'] = predictions
        
        return result_df, predictions
    
    def predict_from_dataframe(self, df):
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        # Preprocess
        X_processed = self.preprocessor.preprocess_for_prediction(df)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['predictions'] = predictions
        
        return result_df, predictions
    
    def generate_shap_explanation(self, df, sample_size=50):
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")
        
        try:
            # Preprocess data
            X_processed = self.preprocessor.preprocess_for_prediction(df)
            
            # Use sample for SHAP calculation
            if len(X_processed) > sample_size:
                sample_indices = np.random.choice(len(X_processed), sample_size, replace=False)
                X_sample = X_processed.iloc[sample_indices] if hasattr(X_processed, 'iloc') else X_processed[sample_indices]
            else:
                X_sample = X_processed
            
            # Create SHAP explainer
            explainer = shap.Explainer(self.model.predict, X_sample)
            shap_values = explainer(X_sample)
            
            return shap_values, X_sample
        except Exception as e:
            print(f"SHAP explanation generation failed: {str(e)}")
            return None, None
    
    def create_shap_summary_plot(self, shap_values, feature_names=None):
        try:
            plt.figure(figsize=(8, 5))
            
            # Set matplotlib backend to Agg for server environments
            import matplotlib
            matplotlib.use('Agg')
            
            # Create SHAP summary plot
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150, 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            plt.clf()  # Clear the figure
            
            return image_base64
        except Exception as e:
            print(f"SHAP summary plot creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()  # Ensure figure is closed even on error
            return None
    
    def create_individual_shap_plot(self, shap_values, sample_index=0):
        try:
            plt.figure(figsize=(8, 5))
            
            # Set matplotlib backend to Agg for server environments
            import matplotlib
            matplotlib.use('Agg')
            
            # Create individual SHAP waterfall plot
            if sample_index < len(shap_values):
                shap.waterfall_plot(shap_values[sample_index], show=False)
            else:
                # Fallback to first sample if index is out of range
                shap.waterfall_plot(shap_values[0], show=False)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150,
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            plt.clf()  # Clear the figure
            
            return image_base64
        except Exception as e:
            print(f"Individual SHAP plot creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()  # Ensure figure is closed even on error
            return None
    
    def create_prediction_distribution_plot(self, predictions):
        try:
            plt.figure(figsize=(8, 5))
            
            # Ensure predictions is a numpy array or list
            if hasattr(predictions, 'tolist'):
                pred_values = predictions.tolist()
            else:
                pred_values = list(predictions)
            
            # Create histogram
            plt.hist(pred_values, bins=min(30, len(pred_values) // 2 + 1), 
                    color='lightblue', alpha=0.7, edgecolor='black')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predictions')
            plt.grid(True, alpha=0.3)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Prediction distribution plot creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_prediction_report(self, file_path=None, df=None):
        if file_path is None and df is None:
            raise ValueError("Either file_path or df must be provided")
        
        # Get predictions
        if file_path:
            result_df, predictions = self.predict_from_file(file_path)
        else:
            result_df, predictions = self.predict_from_dataframe(df)
        
        # Generate SHAP explanation
        shap_values, X_sample = self.generate_shap_explanation(
            df if df is not None else result_df.drop('predictions', axis=1)
        )
        
        # Create plots
        prediction_dist_plot = self.create_prediction_distribution_plot(predictions)
        
        shap_summary = None
        shap_individual = None
        if shap_values is not None:
            feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else None
            shap_summary = self.create_shap_summary_plot(shap_values, feature_names)
            shap_individual = self.create_individual_shap_plot(shap_values, 0)
        
        # Prediction statistics
        pred_stats = {
            'mean': np.mean(predictions),
            'median': np.median(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'count': len(predictions)
        }
        
        report = {
            'predictions': predictions.tolist(),
            'result_dataframe': result_df,
            'statistics': pred_stats,
            'plots': {
                'prediction_distribution': prediction_dist_plot,
                'shap_summary': shap_summary,
                'shap_individual': shap_individual
            }
        }
        
        return report
    
    def save_predictions(self, result_df, output_path):
        if output_path.endswith('.csv'):
            result_df.to_csv(output_path, index=False)
        elif output_path.endswith(('.xlsx', '.xls')):
            result_df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format")
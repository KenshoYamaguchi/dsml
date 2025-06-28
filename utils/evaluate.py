import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import base64
from io import BytesIO
import json

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.predictions = None
        self.shap_values = None
        self.explainer = None
        
    def calculate_metrics(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.predictions)),
            'mae': mean_absolute_error(self.y_test, self.predictions),
            'r2': r2_score(self.y_test, self.predictions),
            'mse': mean_squared_error(self.y_test, self.predictions)
        }
        
        return metrics
    
    def create_prediction_plot(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        try:
            plt.figure(figsize=(8, 5))
            
            # Scatter plot of predictions vs actual
            plt.scatter(self.y_test, self.predictions, alpha=0.6, color='blue', label='Predictions')
            
            # Perfect prediction line
            min_val = min(min(self.y_test), min(self.predictions))
            max_val = max(max(self.y_test), max(self.predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Prediction plot creation failed: {str(e)}")
            return None
    
    def create_residual_plot(self):
        if self.predictions is None:
            self.predictions = self.model.predict(self.X_test)
        
        residuals = self.y_test - self.predictions
        
        try:
            plt.figure(figsize=(8, 5))
            
            # Scatter plot of residuals
            plt.scatter(self.predictions, residuals, alpha=0.6, color='green', label='Residuals')
            
            # Zero line
            plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Residual plot creation failed: {str(e)}")
            return None
    
    def calculate_feature_importance(self):
        if hasattr(self.model, 'feature_importance'):
            importance = self.model.feature_importance(importance_type='gain')
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")
        
        if self.feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        return importance_df
    
    def create_feature_importance_plot(self):
        importance_df = self.calculate_feature_importance()
        
        try:
            plt.figure(figsize=(10, max(5, min(8, len(importance_df) * 0.25))))
            
            # Horizontal bar plot
            plt.barh(importance_df['feature'], importance_df['importance'], color='lightblue')
            
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"Feature importance plot creation failed: {str(e)}")
            return None
    
    def calculate_shap_values(self, sample_size=100):
        try:
            # Use a sample of data for SHAP calculation to improve performance
            if len(self.X_test) > sample_size:
                sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
                X_sample = self.X_test.iloc[sample_indices] if hasattr(self.X_test, 'iloc') else self.X_test[sample_indices]
            else:
                X_sample = self.X_test
            
            # Create explainer
            if hasattr(self.model, 'predict'):
                self.explainer = shap.Explainer(self.model.predict, X_sample)
            else:
                self.explainer = shap.Explainer(self.model, X_sample)
            
            self.shap_values = self.explainer(X_sample)
            
            return self.shap_values
        except Exception as e:
            print(f"SHAP calculation failed: {str(e)}")
            return None
    
    def create_shap_summary_plot(self):
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if self.shap_values is None:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, feature_names=self.feature_names, show=False)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"SHAP summary plot creation failed: {str(e)}")
            return None
    
    def create_shap_waterfall_plot(self, sample_index=0):
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if self.shap_values is None:
            return None
        
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(self.shap_values[sample_index], show=False)
            
            # Convert to base64 for web display
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        except Exception as e:
            print(f"SHAP waterfall plot creation failed: {str(e)}")
            return None
    
    def generate_evaluation_report(self):
        metrics = self.calculate_metrics()
        
        # Create plots (all return base64 images now)
        prediction_plot = self.create_prediction_plot()
        residual_plot = self.create_residual_plot()
        feature_importance_plot = self.create_feature_importance_plot()
        
        # SHAP plots (optional, may fail)
        shap_summary = self.create_shap_summary_plot()
        shap_waterfall = self.create_shap_waterfall_plot()
        
        report = {
            'metrics': metrics,
            'plots': {
                'prediction_plot': prediction_plot,
                'residual_plot': residual_plot,
                'feature_importance_plot': feature_importance_plot,
                'shap_summary': shap_summary,
                'shap_waterfall': shap_waterfall
            }
        }
        
        return report
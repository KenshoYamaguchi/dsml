import lightgbm as lgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

# 関連クラス: DataPreprocessor, ModelEvaluator, Config
# 目的: LightGBMモデルの訓練、ハイパーパラメーターチューニング、モデルの保存・読み込み
# 機能: パラメーターグリッド管理、RandomizedSearchCV、早期停止、特徴重要度算出、モデル評価
class LightGBMTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def get_default_params(self):
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'num_iterations': 300,
            'random_state': 42,
            'verbose': -1
        }
    
    def get_hyperparameter_grid(self):
        return {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9],
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
            'min_child_samples': [10, 20, 30, 50],
            'num_iterations': [100, 300, 500]
        }
    
    def train_with_hyperparameter_tuning(self, X_train, y_train, X_val=None, y_val=None, n_iter=5):
        # Base model for hyperparameter tuning
        base_model = lgb.LGBMRegressor(**self.get_default_params())
        
        # Hyperparameter grid
        param_grid = self.get_hyperparameter_grid()
        
        # Random search
        random_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        
        # Train final model with best parameters
        final_params = self.get_default_params()
        final_params.update(self.best_params)
        
        return self.train_model(X_train, y_train, X_val, y_val, final_params)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, params=None):
        if params is None:
            params = self.get_default_params()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('eval')
        
        # Training parameters
        num_boost_round = params.pop('num_iterations', 300)
        
        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)  # Show progress every 100 iterations
        ]
        
        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, importance_type='gain'):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.feature_importance(importance_type=importance_type)
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Save both the LightGBM model and the best parameters
        model_data = {
            'model': self.model,
            'best_params': self.best_params
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_params = model_data.get('best_params', None)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return {
            'rmse': rmse,
            'predictions': predictions
        }
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.numeric_columns = []
        self.categorical_columns = []
        
    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
    
    def get_column_types(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        return numeric_cols, categorical_cols
    
    def preprocess_for_training(self, df, target_column, feature_columns):
        df_processed = df[feature_columns + [target_column]].copy()
        
        self.numeric_columns, self.categorical_columns = self.get_column_types(
            df_processed.drop(columns=[target_column])
        )
        
        # Remove target column from feature columns if present
        self.numeric_columns = [col for col in self.numeric_columns if col != target_column]
        self.categorical_columns = [col for col in self.categorical_columns if col != target_column]
        
        # Handle missing values
        if self.numeric_columns:
            df_processed[self.numeric_columns] = self.numeric_imputer.fit_transform(
                df_processed[self.numeric_columns]
            )
        
        if self.categorical_columns:
            df_processed[self.categorical_columns] = self.categorical_imputer.fit_transform(
                df_processed[self.categorical_columns]
            )
            
            # Label encode categorical variables
            for col in self.categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                # Ensure encoded values are integers
                df_processed[col] = df_processed[col].astype('int32')
        
        # Scale numeric features
        if self.numeric_columns:
            df_processed[self.numeric_columns] = self.numeric_scaler.fit_transform(
                df_processed[self.numeric_columns]
            )
            # Ensure numeric columns are float
            df_processed[self.numeric_columns] = df_processed[self.numeric_columns].astype('float32')
        
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Debug: Check data types
        print("データ型チェック（学習用）:")
        for col in X.columns:
            print(f"  {col}: {X[col].dtype}")
        
        return X, y
    
    def preprocess_for_prediction(self, df):
        df_processed = df.copy()
        
        # Handle missing values
        if self.numeric_columns:
            df_processed[self.numeric_columns] = self.numeric_imputer.transform(
                df_processed[self.numeric_columns]
            )
        
        if self.categorical_columns:
            df_processed[self.categorical_columns] = self.categorical_imputer.transform(
                df_processed[self.categorical_columns]
            )
            
            # Label encode categorical variables
            for col in self.categorical_columns:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_processed[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        # Replace unseen values with the most frequent value
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_processed[col] = df_processed[col].replace(list(unseen_values), most_frequent)
                    
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                    # Ensure encoded values are integers
                    df_processed[col] = df_processed[col].astype('int32')
        
        # Scale numeric features
        if self.numeric_columns:
            df_processed[self.numeric_columns] = self.numeric_scaler.transform(
                df_processed[self.numeric_columns]
            )
            # Ensure numeric columns are float
            df_processed[self.numeric_columns] = df_processed[self.numeric_columns].astype('float32')
        
        # Debug: Check data types
        print("データ型チェック（予測用）:")
        for col in df_processed.columns:
            print(f"  {col}: {df_processed[col].dtype}")
        
        return df_processed
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def get_basic_statistics(self, df):
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'missing_count': df[col].isnull().sum()
            }
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            stats[col] = {
                'unique_count': df[col].nunique(),
                'top_value': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'missing_count': df[col].isnull().sum()
            }
        
        return stats
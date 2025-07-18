import os

# 関連クラス: DataPreprocessor, LightGBMTrainer, ModelEvaluator, PredictionService
# 目的: アプリケーション全体の設定管理と環境変数の統一管理
# 機能: ファイルアップロード設定、モデル保存パス、Azure設定、データ処理パラメータ等
class Config:
    # Base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Model storage
    MODEL_FOLDER = os.path.join(BASE_DIR, 'model')
    MODEL_FILE = os.path.join(MODEL_FOLDER, 'model.pkl')
    PREPROCESSOR_FILE = os.path.join(MODEL_FOLDER, 'preprocessor.pkl')
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Azure specific settings
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    AZURE_CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME', 'mlapp-files')
    
    # Production mode detection
    IS_PRODUCTION = os.environ.get('WEBSITE_HOSTNAME') is not None
    
    # Data processing settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    HYPERPARAMETER_TUNING_ITERATIONS = 20
    
    # Visualization settings
    PLOT_WIDTH = 800
    PLOT_HEIGHT = 600
    SHAP_SAMPLE_SIZE = 100
    
    @staticmethod
    def init_app(app):
        # Create necessary directories (only if not in production)
        if not Config.IS_PRODUCTION:
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        
        # Set Flask config
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        app.config['SECRET_KEY'] = Config.SECRET_KEY
        app.config['DEBUG'] = Config.DEBUG and not Config.IS_PRODUCTION
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
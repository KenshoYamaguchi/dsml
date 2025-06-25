import os

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
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        
        # Set Flask config
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        app.config['SECRET_KEY'] = Config.SECRET_KEY
        app.config['DEBUG'] = Config.DEBUG
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
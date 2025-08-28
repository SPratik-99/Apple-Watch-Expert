"""
FIXED Configuration - Handles All Deployment Scenarios
Properly detects API keys and environment
"""
import os
from pathlib import Path
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class AppleWatchConfig:
    """Production-ready configuration with proper fallbacks"""
    
    # App settings
    PAGE_TITLE = "Apple Watch Expert"
    PAGE_ICON = "⌚"
    
    # Data paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PDF_DIR = DATA_DIR / "pdfs"
    TXT_DIR = DATA_DIR / "txt" 
    JSON_DIR = DATA_DIR / "prices"
    
    # Vector database
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    COLLECTION_NAME = "apple_watch_docs"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 3
    
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Available Groq models (ordered by preference)
    GROQ_MODELS_TO_TEST = [
        "llama-3.3-70b-specdec",
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
    ]
    
    # Runtime model selection
    SELECTED_GROQ_MODEL = None
    
    @classmethod
    def get_groq_api_key(cls):
        """Get Groq API key with comprehensive fallback"""
        
        # 1. Try Streamlit secrets first (for cloud deployment)
        try:
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                key = str(st.secrets['GROQ_API_KEY']).strip()
                if key and len(key) > 10:  # Valid key should be longer
                    logger.info("Groq API key loaded from Streamlit secrets")
                    return key
        except Exception as e:
            logger.warning(f"Streamlit secrets check failed: {e}")
        
        # 2. Try environment variable
        try:
            key = os.environ.get("GROQ_API_KEY", "").strip()
            if key and len(key) > 10:
                logger.info("Groq API key loaded from environment variable")
                return key
        except Exception as e:
            logger.warning(f"Environment variable check failed: {e}")
        
        # 3. Try .env file (local development)
        try:
            env_file = cls.BASE_DIR / ".env"
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith('GROQ_API_KEY='):
                            key = line.split('=', 1)[1].strip().strip('"\'')
                            if key and len(key) > 10:
                                logger.info("Groq API key loaded from .env file")
                                return key
        except Exception as e:
            logger.warning(f".env file check failed: {e}")
        
        logger.info("No Groq API key found - using fallback models")
        return None
    
    @classmethod
    def is_groq_available(cls) -> bool:
        """Check if Groq API is available and working"""
        key = cls.get_groq_api_key()
        if not key:
            return False
        
        # Test if groq package is available
        try:
            import groq
            return True
        except ImportError:
            logger.warning("Groq package not installed")
            return False
    
    @classmethod
    def test_and_select_best_model(cls) -> str:
        """Test and select the best available Groq model"""
        if not cls.is_groq_available():
            return None
        
        if cls.SELECTED_GROQ_MODEL:
            return cls.SELECTED_GROQ_MODEL
        
        try:
            from groq import Groq
            client = Groq(api_key=cls.get_groq_api_key())
            
            # Test models in order of preference
            for model in cls.GROQ_MODELS_TO_TEST:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Test"}],
                        max_tokens=5,
                        temperature=0.1
                    )
                    
                    if response and response.choices:
                        cls.SELECTED_GROQ_MODEL = model
                        logger.info(f"Selected Groq model: {model}")
                        return model
                        
                except Exception as e:
                    logger.warning(f"Model {model} failed: {e}")
                    continue
            
            logger.warning("No Groq models available")
            return None
            
        except Exception as e:
            logger.error(f"Groq client test failed: {e}")
            return None
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get model information"""
        model_info = {
            "llama-3.3-70b-specdec": {
                "name": "Llama 3.3 70B",
                "description": "Latest high-performance model"
            },
            "llama-3.1-8b-instant": {
                "name": "Llama 3.1 8B", 
                "description": "Fast, efficient model"
            },
            "llama3-8b-8192": {
                "name": "Llama 3 8B",
                "description": "Reliable standard model"
            }
        }
        
        return model_info.get(model_name, {
            "name": model_name,
            "description": "Groq model"
        })
    
    @classmethod
    def ensure_data_directories(cls):
        """Create data directories if they don't exist"""
        try:
            cls.DATA_DIR.mkdir(exist_ok=True)
            cls.PDF_DIR.mkdir(exist_ok=True)
            cls.TXT_DIR.mkdir(exist_ok=True)
            cls.JSON_DIR.mkdir(exist_ok=True)
            cls.CHROMA_DB_DIR.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create data directories: {e}")
    
    @classmethod
    def validate_data_structure(cls) -> dict:
        """Validate data structure"""
        validation = {}
        
        cls.ensure_data_directories()
        
        for folder_name, folder_path in [
            ("pdfs", cls.PDF_DIR),
            ("txt", cls.TXT_DIR), 
            ("prices", cls.JSON_DIR)
        ]:
            exists = folder_path.exists()
            if exists:
                files = [f for f in folder_path.glob("*") if f.is_file() and not f.name.startswith('.')]
                file_count = len(files)
            else:
                file_count = 0
            
            validation[folder_name] = {
                "exists": exists, 
                "files": file_count,
                "status": "ready" if file_count > 0 else "empty"
            }
        
        return validation
    
    @classmethod
    def get_environment(cls) -> str:
        """Detect deployment environment"""
        if 'STREAMLIT_SHARING_MODE' in os.environ or 'streamlit.app' in str(os.environ.get('PWD', '')):
            return "streamlit_cloud"
        elif 'DYNO' in os.environ:
            return "heroku"
        elif 'RENDER' in os.environ:
            return "render"
        elif 'RAILWAY_ENVIRONMENT' in os.environ:
            return "railway"
        else:
            return "local"

# Global config instance
config = AppleWatchConfig()

# Initialize directories
config.ensure_data_directories()
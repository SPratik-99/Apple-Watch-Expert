"""
Updated Configuration for Production Deployment
Handles API keys, environment variables, and local file fallbacks
"""
import os
from pathlib import Path
import streamlit as st

class AppleWatchConfig:
    """Production-ready configuration with environment handling"""
    
    # App settings
    PAGE_TITLE = "Apple Watch Expert"
    PAGE_ICON = "⌚"
    
    # Data paths with fallbacks
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
    
    # Retrieval settings
    RETRIEVAL_K = 3
    
    # API Keys with multiple sources (production ready)
    @classmethod
    def get_groq_api_key(cls):
        """Get Groq API key from multiple sources with fallbacks"""
        # Priority order: Streamlit secrets > Environment variable > None
        
        # 1. Try Streamlit secrets (for cloud deployment)
        try:
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                key = st.secrets['GROQ_API_KEY']
                if key and key.strip():  # Check not empty
                    return key.strip()
        except Exception:
            pass
        
        # 2. Try environment variable (for local development)
        key = os.getenv("GROQ_API_KEY")
        if key and key.strip():
            return key.strip()
        
        # 3. Try .env file (for local development)
        try:
            env_file = cls.BASE_DIR / ".env"
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('GROQ_API_KEY='):
                            key = line.split('=', 1)[1].strip().strip('"\'')
                            if key:
                                return key
        except Exception:
            pass
        
        # 4. No API key available
        return None
    
    @classmethod
    def is_groq_available(cls) -> bool:
        """Check if Groq API key is available"""
        return bool(cls.get_groq_api_key())
    
    # Available Groq models to test (ordered by preference)
    GROQ_MODELS_TO_TEST = [
        "llama-3.3-70b-specdec",    # Newer, potentially better
        "llama-3.1-8b-instant",    # Proven lightweight
        "llama3-8b-8192",          # Fallback option
    ]
    
    # Model selection will be done dynamically
    SELECTED_GROQ_MODEL = None
    
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    @classmethod
    def test_and_select_best_model(cls) -> str:
        """Test available models and select the best one"""
        if not cls.is_groq_available():
            return None
        
        if cls.SELECTED_GROQ_MODEL:
            return cls.SELECTED_GROQ_MODEL
        
        try:
            from groq import Groq
            client = Groq(api_key=cls.get_groq_api_key())
            
            # Test each model with a simple query
            test_prompt = "What is Apple Watch?"
            
            for model in cls.GROQ_MODELS_TO_TEST:
                try:
                    # Test model availability and response
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": test_prompt}],
                        max_tokens=50,
                        temperature=0.1
                    )
                    
                    if response and response.choices:
                        cls.SELECTED_GROQ_MODEL = model
                        print(f"✅ Selected Groq model: {model}")
                        return model
                        
                except Exception as e:
                    print(f"❌ Model {model} failed: {str(e)}")
                    continue
            
            print("⚠️ No Groq models available")
            return None
            
        except Exception as e:
            print(f"⚠️ Groq client failed: {e}")
            return None
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about the selected model"""
        model_info = {
            "llama-3.3-70b-specdec": {
                "name": "Llama 3.3 70B Speculative Decoding",
                "size": "70B parameters",
                "speed": "Fast (speculative decoding)",
                "accuracy": "Very High",
                "description": "Latest Llama model with speculative decoding optimization"
            },
            "llama-3.1-8b-instant": {
                "name": "Llama 3.1 8B Instant", 
                "size": "8B parameters",
                "speed": "Very Fast",
                "accuracy": "High",
                "description": "Lightweight, optimized for speed"
            },
            "llama3-8b-8192": {
                "name": "Llama 3 8B",
                "size": "8B parameters", 
                "speed": "Fast",
                "accuracy": "High",
                "description": "Reliable 8B model"
            }
        }
        
        return model_info.get(model_name, {
            "name": model_name,
            "description": "Unknown model"
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
            print(f"Warning: Could not create data directories: {e}")
    
    @classmethod
    def validate_data_structure(cls) -> dict:
        """Validate data folder structure and return status"""
        validation = {}
        
        # Ensure directories exist first
        cls.ensure_data_directories()
        
        for folder_name, folder_path in [
            ("pdfs", cls.PDF_DIR),
            ("txt", cls.TXT_DIR), 
            ("prices", cls.JSON_DIR)
        ]:
            exists = folder_path.exists()
            if exists:
                # Count files (excluding .gitkeep and hidden files)
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
        # Check for Streamlit Cloud
        if os.getenv('STREAMLIT_SHARING_MODE') or 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
            return "streamlit_cloud"
        
        # Check for other cloud platforms
        if os.getenv('DYNO'):  # Heroku
            return "heroku"
        
        if os.getenv('RENDER'):  # Render
            return "render"
        
        if os.getenv('RAILWAY_ENVIRONMENT'):  # Railway
            return "railway"
        
        # Local development
        return "local"
    
    @classmethod
    def get_app_url(cls) -> str:
        """Get application URL based on environment"""
        env = cls.get_environment()
        
        if env == "streamlit_cloud":
            # Try to get from Streamlit context
            try:
                import streamlit as st
                if hasattr(st, 'get_option'):
                    return st.get_option('server.baseUrlPath') or "streamlit.app"
            except:
                pass
            return "streamlit.app"
        
        elif env == "local":
            return "http://localhost:8501"
        
        else:
            return "deployed_app"

# Global config instance
config = AppleWatchConfig()

# Initialize data directories on import
config.ensure_data_directories()
"""
Production Document Loader
Handles missing files gracefully for deployment scenarios
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json

# Only import PDF processing if available
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None

from config import config

logger = logging.getLogger(__name__)

class Document:
    """Simple document class for production use"""
    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class AppleWatchDocumentLoader:
    """Production-ready document loader with graceful fallbacks"""
    
    def __init__(self):
        self.data_status = config.validate_data_structure()
        self.pdf_available = PDF_AVAILABLE
        
    def load_all_documents(self) -> List[Document]:
        """Load all available documents with error handling"""
        documents = []
        
        try:
            # Load PDF documents
            pdf_docs = self.load_pdf_documents()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
            
            # Load text documents  
            txt_docs = self.load_text_documents()
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} text documents")
            
            # Load JSON documents
            json_docs = self.load_json_documents() 
            documents.extend(json_docs)
            logger.info(f"Loaded {len(json_docs)} JSON documents")
            
            if not documents:
                logger.info("No documents found - creating sample content")
                documents = self._create_sample_documents()
            
            return documents
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            return self._create_sample_documents()
    
    def load_pdf_documents(self) -> List[Document]:
        """Load PDF documents with production error handling"""
        documents = []
        
        if not self.pdf_available:
            logger.warning("PyPDF2 not available - skipping PDF loading")
            return documents
        
        if not config.PDF_DIR.exists():
            logger.info("PDF directory not found - creating directory")
            config.PDF_DIR.mkdir(parents=True, exist_ok=True)
            return documents
        
        try:
            pdf_files = list(config.PDF_DIR.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                try:
                    content = self._extract_pdf_content(pdf_file)
                    if content.strip():
                        # Split into chunks for better processing
                        chunks = self._split_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                        
                        for i, chunk in enumerate(chunks):
                            metadata = {
                                "source": str(pdf_file.name),
                                "file_type": "pdf",
                                "chunk": i,
                                "total_chunks": len(chunks)
                            }
                            documents.append(Document(chunk, metadata))
                    
                except Exception as e:
                    logger.error(f"Failed to process PDF {pdf_file}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
        
        return documents
    
    def _extract_pdf_content(self, pdf_path: Path) -> str:
        """Extract text content from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num} from {pdf_path}: {e}")
                        continue
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            return ""
    
    def load_text_documents(self) -> List[Document]:
        """Load text documents with error handling"""
        documents = []
        
        if not config.TXT_DIR.exists():
            logger.info("Text directory not found - creating directory")
            config.TXT_DIR.mkdir(parents=True, exist_ok=True)
            return documents
        
        try:
            txt_files = list(config.TXT_DIR.glob("*.txt"))
            
            for txt_file in txt_files:
                try:
                    content = self._read_text_file(txt_file)
                    if content.strip():
                        # Split into chunks
                        chunks = self._split_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                        
                        for i, chunk in enumerate(chunks):
                            metadata = {
                                "source": str(txt_file.name),
                                "file_type": "txt",
                                "chunk": i,
                                "total_chunks": len(chunks)
                            }
                            documents.append(Document(chunk, metadata))
                    
                except Exception as e:
                    logger.error(f"Failed to process text file {txt_file}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Text file loading failed: {e}")
        
        return documents
    
    def _read_text_file(self, txt_path: Path) -> str:
        """Read text file with encoding detection"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(txt_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Failed to read {txt_path} with {encoding}: {e}")
                continue
        
        logger.error(f"Could not read {txt_path} with any encoding")
        return ""
    
    def load_json_documents(self) -> List[Document]:
        """Load JSON documents with error handling"""
        documents = []
        
        if not config.JSON_DIR.exists():
            logger.info("JSON directory not found - creating directory")
            config.JSON_DIR.mkdir(parents=True, exist_ok=True)
            return documents
        
        try:
            json_files = list(config.JSON_DIR.glob("*.json"))
            
            for json_file in json_files:
                try:
                    content = self._process_json_file(json_file)
                    if content.strip():
                        metadata = {
                            "source": str(json_file.name),
                            "file_type": "json"
                        }
                        documents.append(Document(content, metadata))
                    
                except Exception as e:
                    logger.error(f"Failed to process JSON file {json_file}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"JSON file loading failed: {e}")
        
        return documents
    
    def _process_json_file(self, json_path: Path) -> str:
        """Process JSON file into readable text"""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Convert JSON to readable text format
            if isinstance(data, dict):
                return self._json_dict_to_text(data)
            elif isinstance(data, list):
                return self._json_list_to_text(data)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"JSON processing failed for {json_path}: {e}")
            return ""
    
    def _json_dict_to_text(self, data: dict, prefix: str = "") -> str:
        """Convert JSON dictionary to readable text"""
        text_parts = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                text_parts.append(f"{prefix}{key}:")
                text_parts.append(self._json_dict_to_text(value, prefix + "  "))
            elif isinstance(value, list):
                text_parts.append(f"{prefix}{key}: {', '.join(map(str, value))}")
            else:
                text_parts.append(f"{prefix}{key}: {value}")
        
        return "\n".join(text_parts)
    
    def _json_list_to_text(self, data: list) -> str:
        """Convert JSON list to readable text"""
        text_parts = []
        
        for i, item in enumerate(data):
            if isinstance(item, dict):
                text_parts.append(f"Item {i + 1}:")
                text_parts.append(self._json_dict_to_text(item, "  "))
            else:
                text_parts.append(f"Item {i + 1}: {item}")
        
        return "\n".join(text_parts)
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for char in ['. ', '.\n', '? ', '! ']:
                    last_sentence = text.rfind(char, start, end)
                    if last_sentence > start:
                        end = last_sentence + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents when no files are available"""
        sample_docs = [
            Document(
                "Apple Watch SE (2nd generation) is the most affordable Apple Watch, starting at ₹24,900 for 40mm GPS model. It features S8 SiP chip, heart rate monitoring, sleep tracking, crash detection, fall detection, and GPS. The SE lacks Always-On display, ECG, Blood Oxygen monitoring, and Double Tap gesture compared to Series 9.",
                {"source": "sample_se_info.txt", "file_type": "txt", "chunk": 0}
            ),
            Document(
                "Apple Watch Series 9 starts at ₹41,900 for 41mm GPS model. It includes S9 SiP chip (60% faster than SE), Always-On Retina display with 2000 nits brightness, Double Tap gesture control, ECG app, Blood Oxygen monitoring, temperature sensing, and on-device Siri processing. Available in 41mm and 45mm sizes.",
                {"source": "sample_series9_info.txt", "file_type": "txt", "chunk": 0}
            ),
            Document(
                "Apple Watch Ultra 2 is priced at ₹89,900 for 49mm cellular model. Features aerospace-grade titanium case, largest Apple Watch display, Action Button, precision dual-frequency GPS, 100m water resistance, 36-hour battery life (72 hours in Low Power Mode), and 86dB emergency siren. Designed for extreme sports and outdoor adventures.",
                {"source": "sample_ultra_info.txt", "file_type": "txt", "chunk": 0}
            ),
            Document(
                "Apple Watch charging troubleshooting: Clean charging contacts with soft cloth, use original Apple charging cable, check power adapter (5W minimum), ensure proper magnetic connection. For frozen screen: hold side button + Digital Crown for 10 seconds until Apple logo appears. Battery optimization: disable Always-On display, reduce wake time, close unnecessary apps.",
                {"source": "sample_troubleshooting.txt", "file_type": "txt", "chunk": 0}
            ),
            Document(
                '{"apple_watch_pricing": {"SE_40mm_GPS": 24900, "SE_44mm_GPS": 28900, "Series9_41mm_GPS": 41900, "Series9_45mm_GPS": 44900, "Ultra2_49mm": 89900}, "health_features": {"SE": ["heart_rate", "sleep_tracking"], "Series9": ["heart_rate", "sleep_tracking", "ECG", "blood_oxygen", "temperature"], "Ultra2": ["all_series9_features", "depth_gauge", "water_temperature"]}}',
                {"source": "sample_pricing.json", "file_type": "json"}
            )
        ]
        
        logger.info("Created sample documents for demonstration")
        return sample_docs
    
    def get_loading_status(self) -> Dict[str, Any]:
        """Get document loading status"""
        return {
            "pdf_support": self.pdf_available,
            "data_status": self.data_status,
            "directories_exist": {
                "pdfs": config.PDF_DIR.exists(),
                "txt": config.TXT_DIR.exists(),
                "json": config.JSON_DIR.exists()
            }
        }
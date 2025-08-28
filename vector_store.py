"""
Production Vector Store Manager  
Handles missing data files gracefully for deployment
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import streamlit as st

# Only import if available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from document_loader import AppleWatchDocumentLoader
from config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Production-ready vector store with graceful fallbacks"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embeddings_model = None
        self.is_initialized = False
        self.documents_available = False
        
        # Check if we have the required libraries
        self.chromadb_available = CHROMADB_AVAILABLE
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        
        # Validate data structure
        self.data_status = config.validate_data_structure()
        
    def initialize_vector_store(self) -> bool:
        """Initialize vector store with production error handling"""
        try:
            # Check if we have documents to process
            total_files = sum(
                self.data_status.get(folder, {}).get("files", 0) 
                for folder in ["pdfs", "txt", "prices"]
            )
            
            if total_files == 0:
                logger.info("No documents found - vector store will use fallback responses")
                self.is_initialized = True
                self.documents_available = False
                return True
            
            # Check if required libraries are available
            if not self.chromadb_available:
                logger.warning("ChromaDB not available - using fallback responses")
                self.is_initialized = True
                self.documents_available = False
                return True
            
            if not self.embeddings_available:
                logger.warning("Sentence transformers not available - using fallback responses")
                self.is_initialized = True
                self.documents_available = False
                return True
            
            # Initialize ChromaDB
            try:
                # Use persistent storage in data directory
                persist_directory = str(config.CHROMA_DB_DIR)
                
                # Ensure directory exists
                config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
                
                self.client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # Get or create collection
                try:
                    self.collection = self.client.get_collection(config.COLLECTION_NAME)
                    logger.info("Loaded existing vector store")
                except:
                    # Create new collection
                    self.collection = self.client.create_collection(
                        name=config.COLLECTION_NAME,
                        metadata={"description": "Apple Watch documents and specifications"}
                    )
                    logger.info("Created new vector store")
                    
                    # Load documents if collection is empty
                    if self.collection.count() == 0:
                        self._load_documents()
                
            except Exception as e:
                logger.error(f"ChromaDB initialization failed: {e}")
                self.is_initialized = True
                self.documents_available = False
                return True
            
            # Initialize embeddings model
            try:
                self.embeddings_model = SentenceTransformer(config.EMBEDDING_MODEL)
                logger.info("Embeddings model loaded successfully")
            except Exception as e:
                logger.error(f"Embeddings model failed: {e}")
                self.is_initialized = True
                self.documents_available = False
                return True
            
            self.is_initialized = True
            self.documents_available = True
            logger.info("Vector store initialized successfully with documents")
            return True
            
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            # Graceful fallback - system still works
            self.is_initialized = True
            self.documents_available = False
            return True
    
    def _load_documents(self):
        """Load documents with production error handling"""
        try:
            document_loader = AppleWatchDocumentLoader()
            documents = document_loader.load_all_documents()
            
            if not documents:
                logger.info("No documents to load")
                return
            
            # Process documents in batches
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Generate embeddings
                texts = [doc.page_content for doc in batch]
                embeddings = self.embeddings_model.encode(texts, convert_to_tensor=False)
                
                # Prepare metadata
                metadatas = []
                ids = []
                
                for j, doc in enumerate(batch):
                    doc_id = f"doc_{i+j}_{hash(doc.page_content) % 10000}"
                    ids.append(doc_id)
                    
                    metadata = {"source": "apple_watch_document"}
                    if hasattr(doc, 'metadata') and doc.metadata:
                        metadata.update(doc.metadata)
                    metadatas.append(metadata)
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Loaded {len(documents)} documents into vector store")
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            # Don't fail - just continue without documents
    
    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """Perform similarity search with fallback"""
        try:
            if not self.is_ready():
                return []
            
            if not self.documents_available:
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_tensor=False)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(k, self.collection.count())
            )
            
            # Convert results to document-like objects
            documents = []
            if results['documents'] and results['documents'][0]:
                for doc_text, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    # Create simple document object
                    class SimpleDoc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta
                    
                    documents.append(SimpleDoc(doc_text, metadata))
            
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        return self.is_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get vector store status"""
        status = {
            "initialized": self.is_initialized,
            "documents_available": self.documents_available,
            "chromadb_available": self.chromadb_available,
            "embeddings_available": self.embeddings_available,
        }
        
        if self.collection:
            try:
                status["document_count"] = self.collection.count()
            except:
                status["document_count"] = 0
        else:
            status["document_count"] = 0
        
        return status
    
    def reset(self):
        """Reset vector store"""
        try:
            if self.client and self.collection:
                self.client.delete_collection(config.COLLECTION_NAME)
                self.collection = None
                logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Vector store reset failed: {e}")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None):
        """Add single document to vector store"""
        try:
            if not self.is_ready() or not self.documents_available:
                return False
            
            # Generate embedding
            embedding = self.embeddings_model.encode([text], convert_to_tensor=False)
            
            # Create ID
            doc_id = f"manual_doc_{hash(text) % 10000}"
            
            # Prepare metadata
            if metadata is None:
                metadata = {"source": "manual_addition"}
            
            # Add to collection
            self.collection.add(
                embeddings=embedding.tolist(),
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Add document failed: {e}")
            return False
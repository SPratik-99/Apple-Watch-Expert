"""
Simple and Robust Vector Store Manager
Handles all import errors gracefully without syntax issues
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Simple, crash-proof vector store manager"""
    
    def __init__(self):
        self.is_initialized = False
        self.documents_available = False
        self.error_message = ""
        
        # Try to import optional dependencies
        self.chromadb_available = self._test_chromadb()
        self.embeddings_available = self._test_embeddings()
        
        logger.info(f"VectorStore: ChromaDB={self.chromadb_available}, Embeddings={self.embeddings_available}")
    
    def _test_chromadb(self) -> bool:
        """Test if ChromaDB is available"""
        try:
            import chromadb
            return True
        except Exception as e:
            self.error_message = f"ChromaDB not available: {str(e)[:100]}"
            logger.warning(self.error_message)
            return False
    
    def _test_embeddings(self) -> bool:
        """Test if sentence transformers is available"""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except Exception as e:
            logger.warning(f"Sentence transformers not available: {e}")
            return False
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store - always succeeds"""
        try:
            self.is_initialized = True
            
            if not self.chromadb_available or not self.embeddings_available:
                logger.info("Vector components not available - using fallback mode")
                self.documents_available = False
                return True
            
            # Try full initialization only if components available
            success = self._try_full_initialization()
            self.documents_available = success
            
            logger.info(f"Vector store initialized: documents_available={self.documents_available}")
            return True
            
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
            self.is_initialized = True  # Still mark as initialized
            self.documents_available = False
            return True
    
    def _try_full_initialization(self) -> bool:
        """Try to set up full vector store functionality"""
        try:
            # Import here to avoid top-level import errors
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
            
            # Set up directory
            persist_dir = Path("./data/chroma_db")
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            
            # Get or create collection
            try:
                collection = client.get_collection("apple_watch_docs")
                logger.info("Loaded existing collection")
            except:
                collection = client.create_collection("apple_watch_docs")
                logger.info("Created new collection")
            
            # Initialize embeddings
            embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Store for later use
            self.client = client
            self.collection = collection
            self.embeddings_model = embeddings_model
            
            # Add sample documents if collection is empty
            if collection.count() == 0:
                self._add_sample_documents()
            
            return True
            
        except Exception as e:
            logger.error(f"Full initialization failed: {e}")
            return False
    
    def _add_sample_documents(self):
        """Add sample documents for demonstration"""
        try:
            sample_docs = [
                "Apple Watch SE (2nd generation) starts at ₹24,900 for 40mm GPS model. Features S8 chip, heart rate monitoring, sleep tracking, crash detection, GPS. Lacks Always-On display, ECG, Blood Oxygen.",
                "Apple Watch Series 9 starts at ₹41,900 for 41mm GPS model. Features S9 chip (60% faster), Always-On display, Double Tap gesture, ECG app, Blood Oxygen monitoring, temperature sensing.",
                "Apple Watch Ultra 2 costs ₹89,900. Features titanium case, Action Button, precision GPS, 100m water resistance, 36-hour battery, 86dB siren. For extreme sports and adventures.",
                "Apple Watch charging fix: Clean contacts with soft cloth, use original cable, check power adapter, ensure magnetic connection. For frozen screen: hold buttons for 10 seconds."
            ]
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode(sample_docs)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=sample_docs,
                ids=[f"sample_{i}" for i in range(len(sample_docs))],
                metadatas=[{"source": "sample", "type": "demo"} for _ in sample_docs]
            )
            
            logger.info("Added sample documents to collection")
            
        except Exception as e:
            logger.error(f"Sample documents failed: {e}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """Perform similarity search with error handling"""
        try:
            if not self.is_ready() or not self.documents_available:
                return []
            
            if not hasattr(self, 'collection') or not hasattr(self, 'embeddings_model'):
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])
            
            # Search
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(k, self.collection.count())
            )
            
            # Convert to simple document objects
            documents = []
            if results.get('documents') and results['documents'][0]:
                for doc_text, metadata in zip(results['documents'][0], results.get('metadatas', [[]])[0]):
                    class SimpleDoc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta or {}
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
            "error_message": self.error_message
        }
        
        # Try to get document count
        try:
            if hasattr(self, 'collection') and self.collection:
                status["document_count"] = self.collection.count()
            else:
                status["document_count"] = 0
        except:
            status["document_count"] = 0
        
        return status
    
    def reset(self):
        """Reset vector store"""
        try:
            if hasattr(self, 'client') and hasattr(self, 'collection'):
                self.client.delete_collection("apple_watch_docs")
            logger.info("Vector store reset")
        except Exception as e:
            logger.error(f"Reset failed: {e}")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """Add document to vector store"""
        try:
            if not self.documents_available or not hasattr(self, 'collection'):
                return False
            
            embedding = self.embeddings_model.encode([text])
            doc_id = f"doc_{hash(text) % 10000}"
            
            self.collection.add(
                embeddings=embedding.tolist(),
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata or {"source": "manual"}]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Add document failed: {e}")
            return False

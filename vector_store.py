"""
Crash-Resistant Vector Store Manager  
Handles ChromaDB import and runtime errors gracefully
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

# Try to import optional dependencies with graceful fallbacks
CHROMADB_AVAILABLE = False
EMBEDDINGS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB imported successfully")
except Exception as e:
    logger.warning(f"ChromaDB not available: {e}")
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    logger.info("Sentence transformers imported successfully")
except Exception as e:
    logger.warning(f"Sentence transformers not available: {e}")
    SentenceTransformer = None

# Safe imports
try:
    from document_loader import AppleWatchDocumentLoader
    DOCUMENT_LOADER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Document loader not available: {e}")
    DOCUMENT_LOADER_AVAILABLE = False

try:
    from config import config
    CONFIG_AVAILABLE = True
except Exception as e:
    logger.warning(f"Config not available: {e}")
    CONFIG_AVAILABLE = False

class VectorStoreManager:
    """Crash-resistant vector store with comprehensive error handling"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embeddings_model = None
        self.is_initialized = False
        self.documents_available = False
        self.error_state = False
        
        # Track availability of components
        self.chromadb_available = CHROMADB_AVAILABLE
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        self.document_loader_available = DOCUMENT_LOADER_AVAILABLE
        self.config_available = CONFIG_AVAILABLE
        
        # Set up safe defaults
        self.data_status = {"pdfs": {"files": 0}, "txt": {"files": 0}, "prices": {"files": 0}}
        
        # Try to get config data status if available
        if CONFIG_AVAILABLE:
            try:
                self.data_status = config.validate_data_structure()
            except Exception as e:
                logger.warning(f"Could not validate data structure: {e}")
        
        logger.info(f"VectorStore initialized - ChromaDB: {self.chromadb_available}, Embeddings: {self.embeddings_available}")
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store with maximum error resilience"""
        try:
            # Always return True - system should work regardless
            self.is_initialized = True
            
            # Check if we have documents to process
            total_files = sum(
                self.data_status.get(folder, {}).get("files", 0) 
                for folder in ["pdfs", "txt", "prices"]
            )
            
            if total_files == 0:
                logger.info("No documents found - using fallback responses")
                self.documents_available = False
                return True
            
            # Check if required libraries are available
            if not self.chromadb_available:
                logger.info("ChromaDB not available - using fallback responses")
                self.documents_available = False
                return True
            
            if not self.embeddings_available:
                logger.info("Embeddings not available - using fallback responses")
                self.documents_available = False
                return True
            
            # Try to initialize ChromaDB with comprehensive error handling
            success = self._safe_initialize_chromadb()
            if success:
                self.documents_available = True
                logger.info("ChromaDB initialized successfully with documents")
            else:
                self.documents_available = False
                logger.info("ChromaDB initialization failed - using fallbacks")
            
            return True  # Always return True
            
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
            # Even if initialization fails completely, don't crash the app
            self.is_initialized = True
            self.documents_available = False
            self.error_state = True
            return True
    
    def _safe_initialize_chromadb(self) -> bool:
        """Safely initialize ChromaDB with error handling"""
        try:
            # Set up directories safely
            if CONFIG_AVAILABLE:
                persist_directory = str(config.CHROMA_DB_DIR)
                config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
            else:
                # Use default directory
                persist_directory = "./data/chroma_db"
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            collection_name = config.COLLECTION_NAME if CONFIG_AVAILABLE else "apple_watch_docs"
            
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info("Loaded existing ChromaDB collection")
            except:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Apple Watch documents and specifications"}
                )
                logger.info("Created new ChromaDB collection")
                
                # Try to load documents if collection is empty
                if self.collection.count() == 0:
                    self._safe_load_documents()
            
            # Initialize embeddings model
            embedding_model = config.EMBEDDING_MODEL if CONFIG_AVAILABLE else "all-MiniLM-L6-v2"
            self.embeddings_model = SentenceTransformer(embedding_model)
            logger.info("Embeddings model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            # Clean up any partial initialization
            self.client = None
            self.collection = None
            self.embeddings_model = None
            return False
    
    def _safe_load_documents(self):
        """Safely load documents with error handling"""
        try:
            if not DOCUMENT_LOADER_AVAILABLE:
                logger.warning("Document loader not available - using sample documents")
                self._create_sample_documents()
                return
            
            document_loader = AppleWatchDocumentLoader()
            documents = document_loader.load_all_documents()
            
            if not documents:
                logger.info("No documents loaded - creating samples")
                self._create_sample_documents()
                return
            
            # Process documents in small batches to avoid memory issues
            batch_size = 5
            processed_count = 0
            
            for i in range(0, len(documents), batch_size):
                try:
                    batch = documents[i:i + batch_size]
                    
                    # Generate embeddings
                    texts = [doc.page_content for doc in batch]
                    embeddings = self.embeddings_model.encode(texts, convert_to_tensor=False)
                    
                    # Prepare metadata and IDs
                    metadatas = []
                    ids = []
                    
                    for j, doc in enumerate(batch):
                        doc_id = f"doc_{i+j}_{hash(doc.page_content[:100]) % 10000}"
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
                    
                    processed_count += len(batch)
                    logger.info(f"Processed {processed_count}/{len(documents)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to process document batch {i}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {processed_count} documents into ChromaDB")
            
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            # Try to create sample documents as fallback
            try:
                self._create_sample_documents()
            except:
                logger.error("Could not create sample documents either")
    
    def _create_sample_documents(self):
        """Create sample documents for demonstration"""
        try:
            if not self.collection or not self.embeddings_model:
                return
            
            sample_docs = [
                "Apple Watch SE (2nd generation) starts at ₹24,900 for 40mm GPS model. Features S8 SiP chip, heart rate monitoring, sleep tracking, crash detection, fall detection, and GPS. Lacks Always-On display, ECG, Blood Oxygen monitoring compared to Series 9.",
                "Apple Watch Series 9 starts at ₹41,900 for 41mm GPS model. Includes S9 SiP chip (60% faster), Always-On Retina display, Double Tap gesture, ECG app, Blood Oxygen monitoring, temperature sensing, on-device Siri processing.",
                "Apple Watch Ultra 2 costs ₹89,900. Features titanium case, Action Button, precision GPS, 100m water resistance, 36-hour battery life, 86dB emergency siren. Designed for extreme sports and outdoor adventures.",
                "Apple Watch charging troubleshooting: Clean contacts, use original cable, check power adapter, ensure magnetic connection. For frozen screen: hold side button + Digital Crown for 10 seconds.",
            ]
            
            # Create embeddings and add to collection
            embeddings = self.embeddings_model.encode(sample_docs, convert_to_tensor=False)
            
            ids = [f"sample_{i}" for i in range(len(sample_docs))]
            metadatas = [{"source": "sample_document", "type": "demo"} for _ in sample_docs]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=sample_docs,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("Created sample documents for demonstration")
            
        except Exception as e:
            logger.error(f"Sample document creation failed: {e}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """Perform similarity search with comprehensive error handling"""
        try:
            if not self.is_ready():
                logger.info("Vector store not ready - returning empty results")
                return []
            
            if not self.documents_available or self.error_state:
                logger.info("No documents available - returning empty results")
                return []
            
            if not self.collection or not self.embeddings_model:
                logger.info("ChromaDB components not available - returning empty results")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_tensor=False)
            
            # Search collection
            max_results = min(k, self.collection.count()) if self.collection.count() > 0 else 0
            if max_results == 0:
                return []
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max_results
            )
            
            # Convert results to document-like objects
            documents = []
            if results.get('documents') and results['documents'][0]:
                for doc_text, metadata in zip(results['documents'][0], results.get('metadatas', [[]])[0]):
                    # Create simple document object
                    class SimpleDoc:
                        def __init__(self, content, meta):
                            self.page_content = content
                            self.metadata = meta or {}
                    
                    documents.append(SimpleDoc(doc_text, metadata))
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if vector store is ready - always returns True for app stability"""
        return self.is_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive vector store status"""
        status = {
            "initialized": self.is_initialized,
            "documents_available": self.documents_available,
            "chromadb_available": self.chromadb_available,
            "embeddings_available": self.embeddings_available,
            "document_loader_available": self.document_loader_available,
            "config_available": self.config_available,
            "error_state": self.error_state,
        }
        
        # Try to get document count safely
        try:
            if self.collection:
                status["document_count"] = self.collection.count()
            else:
                status["document_count"] = 0
        except:
            status["document_count"] = 0
        
        return status
    
    def reset(self):
        """Safely reset vector store"""
        try:
            if self.client and self.collection:
                collection_name = config.COLLECTION_NAME if CONFIG_AVAILABLE else "apple_watch_docs"
                self.client.delete_collection(collection_name)
                self.collection = None
                logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Vector store reset failed: {e}")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> bool:
        """Safely add single document to vector store"""
        try:
            if not self.is_ready() or not self.documents_available:
                logger.info("Vector store not ready for adding documents")
                return False
            
            if not self.collection or not self.embeddings_model:
                logger.info("ChromaDB components not available")
                return False
            
            # Generate embedding
            embedding = self.embeddings_model.encode([text], convert_to_tensor=False)
            
            # Create ID
            doc_id = f"manual_doc_{hash(text[:100]) % 10000}"
            
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
            
            logger.info("Document added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Add document failed: {e}")
            return False
    
    def health_check(self) -> Dict[str, str]:
        """Perform health check on vector store components"""
        health = {}
        
        # Test ChromaDB
        try:
            if self.client and self.collection:
                count = self.collection.count()
                health["chromadb"] = f"OK ({count} docs)"
            else:
                health["chromadb"] = "Not available"
        except Exception as e:
            health["chromadb"] = f"Error: {str(e)[:50]}"
        
        # Test embeddings
        try:
            if self.embeddings_model:
                test_embedding = self.embeddings_model.encode(["test"], convert_to_tensor=False)
                health["embeddings"] = f"OK (dim={len(test_embedding[0])})"
            else:
                health["embeddings"] = "Not available"
        except Exception as e:
            health["embeddings"] = f"Error: {str(e)[:50]}"
        
        return health
            
        except Exception as e:
            logger.error(f"Add document failed: {e}")
            return False

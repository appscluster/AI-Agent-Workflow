"""
Document embeddings utilities.
"""
import os
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbeddings:
    """
    Class for handling document embeddings.
    """
    
    def __init__(
        self,
        persist_dir: str,
        openai_api_key: Optional[str] = None,
        collection_name: str = "document_collection"  # Added as optional parameter with default
    ):
        """
        Initialize document embeddings.
        
        Args:
            persist_dir: Directory to persist embeddings
            openai_api_key: OpenAI API key
            collection_name: Name of the collection for vector store
        """
        self.persist_dir = persist_dir
        self.openai_api_key = openai_api_key
        self.collection_name = collection_name
        self.embedding_store = None
        self.indexed = False
        self.documents = []
        self.document_metadata = {}
        
        # Initialize embedding store
        self._initialize_embedding_store()
    
    def _initialize_embedding_store(self):
        """
        Initialize embedding store.
        """
        try:
            # LlamaIndex imports
            from llama_index import ServiceContext
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.storage.storage_context import StorageContext
            from llama_index.vector_stores.simple import SimpleVectorStore
            from llama_index.indices.vector_store import VectorStoreIndex
            
            embedding_model = OpenAIEmbedding(api_key=self.openai_api_key)
            
            # Create service context
            service_context = ServiceContext.from_defaults(embed_model=embedding_model)
            
            # Create vector store and storage context
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create empty index
            try:
                self.index = VectorStoreIndex.from_documents(
                    [],
                    service_context=service_context,
                    storage_context=storage_context
                )
            except AttributeError:
                # Fallback if from_documents not available
                self.index = VectorStoreIndex(
                    [],
                    service_context=service_context,
                    storage_context=storage_context
                )
            
            self.has_llama_index = True
            logger.info("Initialized embedding store with llama_index")
            
        except ImportError as e:
            logger.error(f"Error initializing embedding store with llama_index: {e}")
            logger.warning("Continuing without embeddings support")
            self.has_llama_index = False
            self.embedding_store = None
        except Exception as e:
            logger.error(f"Unexpected error initializing embedding store: {e}")
            logger.warning("Continuing without embeddings support")
            self.has_llama_index = False
            self.embedding_store = None
    
    def process_document(self, document_data: Dict[str, Any]):
        """
        Process document data and create embeddings.
        
        Args:
            document_data: Document data with chunks and metadata
        """
        try:
            chunks = document_data.get("chunks", [])
            metadata = document_data.get("metadata", {})
            
            if not chunks:
                logger.warning("No chunks found in document data")
                return
            
            logger.info(f"Processing {len(chunks)} chunks for embeddings")
            
            # Store documents for later indexing
            self.documents = chunks
            self.document_metadata = metadata
            
            # Index documents
            self.index_documents(chunks, [metadata] * len(chunks))
            
        except Exception as e:
            logger.error(f"Error processing document for embeddings: {e}")
            logger.warning(f"Details: {str(e)}")
    
    def index_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Index documents in the embedding store.
        """
        try:
            if not documents:
                logger.warning("No documents to index")
                return
            if metadata is None:
                metadata = [{} for _ in documents]
            if len(metadata) != len(documents):
                metadata = [metadata[0]] * len(documents)
            logger.info(f"Indexing {len(documents)} documents")
            # Use llama_index exclusively
            if getattr(self, 'has_llama_index', False) and hasattr(self, 'index'):
                try:
                    from llama_index import Document
                    llama_documents = [Document(text=chunk, metadata=meta) for chunk, meta in zip(documents, metadata)]
                    self.index.refresh_ref_docs(llama_documents)
                    logger.info("Documents indexed with llama_index")
                except Exception as e:
                    logger.error(f"Error indexing with llama_index: {e}")
                    self.documents = documents
                    self.document_metadata = metadata[0] if metadata else {}
            else:
                logger.warning("Llama_index not available, storing documents locally")
                self.documents = documents
                self.document_metadata = metadata[0] if metadata else {}
            self.indexed = True
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            self.documents = documents
            self.document_metadata = metadata[0] if metadata else {}
            self.indexed = True  # Consider it indexed even if just stored locally
    
    def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Get relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to return
            
        Returns:
            List of relevant chunks
        """
        # Ensure indexed
        if not self.indexed:
            if self.documents:
                logger.warning("Documents not indexed. Attempting to index now.")
                self.index_documents(self.documents, [self.document_metadata] * len(self.documents))
            else:
                logger.warning("No documents indexed and no documents to index.")
                return []
        # Use llama_index exclusively
        try:
            if getattr(self, 'has_llama_index', False) and hasattr(self, 'index'):
                query_engine = self.index.as_query_engine()
                response = query_engine.query(query)
                if hasattr(response, 'source_nodes'):
                    return [str(node.text) for node in response.source_nodes[:top_k]]
                logger.warning("Query response has no source_nodes attribute")
            # Fallback to simple stored documents
            return self.documents[:top_k] if self.documents else []
        except Exception as e:
            logger.error(f"Error querying llama_index: {e}")
            return self.documents[:top_k] if self.documents else []
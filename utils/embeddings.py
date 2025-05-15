"""
Vector embeddings utilities for document retrieval.
"""
import os
from typing import List, Dict, Any, Optional
import numpy as np

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.schema import Document
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from chromadb.config import Settings

class DocumentEmbeddings:
    """
    Manages document embeddings for semantic search and retrieval.
    Creates and maintains vector embeddings of document chunks
    for efficient semantic retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "document_collection",
        persist_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize embeddings manager.
        
        Args:
            collection_name: Name of the vector collection
            persist_dir: Directory to persist embeddings (optional)
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            dimensions=1536,  # Default for OpenAI embeddings
            embed_batch_size=10  # Process 10 documents at a time
        )
        
        # Initialize vector store
        self.chroma_client = None
        self.vector_store = None
        self.storage_context = None
        self.index = None
        
        # Setup vector store
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """
        Set up the vector store for document embeddings.
        """
        # Create persistent directory if specified
        if self.persist_dir and not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        
        # Initialize Chroma client with persistence
        import chromadb
        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=self.persist_dir,
                anonymized_telemetry=False
            ) if self.persist_dir else Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Create vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
    
    def index_documents(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Index documents in the vector store.
        
        Args:
            documents: List of document chunks to index
            
        Returns:
            VectorStoreIndex: The created index
        """
        # Create vector store index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )
        
        # Persist if directory specified
        if self.persist_dir:
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        
        return self.index
    
    def retrieve_relevant_context(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant document chunks
        """
        if not self.index:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Create retriever
        retriever = self.index.as_retriever(
            similarity_top_k=top_k
        )
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Filter by similarity threshold if specified
        if similarity_threshold > 0:
            nodes = [node for node in nodes if node.score >= similarity_threshold]
        
        return nodes
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5  # Weight between semantic (1.0) and keyword (0.0)
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            alpha: Weight between semantic (1.0) and keyword (0.0) search
            
        Returns:
            List of relevant document chunks
        """
        if not self.index:
            raise ValueError("No documents indexed. Call index_documents first.")
        
        # Create hybrid retriever
        from llama_index.retrievers import QueryFusionRetriever
        
        # Get vector retriever
        vector_retriever = self.index.as_retriever(
            similarity_top_k=top_k
        )
        
        # Get keyword retriever (if alpha < 1.0)
        keyword_retriever = None
        if alpha < 1.0:
            from llama_index.retrievers import BM25Retriever
            keyword_retriever = BM25Retriever.from_defaults(
                docstore=self.index.docstore,
                similarity_top_k=top_k
            )
        
        # If using pure vector search
        if alpha >= 1.0 or not keyword_retriever:
            return vector_retriever.retrieve(query)
        
        # Create fusion retriever
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[alpha, 1.0 - alpha],
            similarity_top_k=top_k,
            num_queries=1  # Just use the original query
        )
        
        # Retrieve nodes
        return fusion_retriever.retrieve(query)
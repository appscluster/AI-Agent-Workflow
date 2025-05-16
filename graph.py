"""
LlamaIndex graph orchestration for document question answering workflow.
"""
from typing import Dict, Any, List, Tuple, Callable, Optional
import os

import logging
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.answer_generator import AnswerGeneratorAgent
from utils.embeddings import DocumentEmbeddings
from utils.memory import ConversationMemory
# Removed import of DocumentKnowledgeGraph

# Optional: Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAWorkflowGraph:
    """
    Graph-based workflow orchestration for document question answering.
    Coordinates the different agents and manages the workflow state
    to process user questions and generate responses.
    """
    
    def __init__(
        self,
        document_path: str,
        persist_dir: Optional[str] = None,
        use_knowledge_graph: bool = True,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the workflow graph.
        
        Args:
            document_path: Path to the document
            persist_dir: Directory to persist embeddings and graph data
            use_knowledge_graph: Whether to use knowledge graph
            openai_api_key: OpenAI API key
        """
        # Set up document path and persistence
        self.document_path = document_path
        self.persist_dir = persist_dir or os.path.join(os.getcwd(), "data")
        self.use_knowledge_graph = use_knowledge_graph
        self.openai_api_key = openai_api_key  # Store API key
        
        # Initialize default attributes to prevent AttributeError
        self.embeddings = None
        self.embeddings_manager = None  # Add this attribute
        self.knowledge_graph = None
        self.document_chunks = []
        self.document_metadata = {}
        self.document_text = ""
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Configure LlamaIndex settings
        if openai_api_key:
            Settings.openai_api_key = openai_api_key
        
        # Set default LLM and embedding model
        Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
        # Use text-embedding-ada-002 which is more widely available
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        
        # Initialize conversation memory
        self.memory = ConversationMemory()
        
        # Initialize document processors and agents
        self._initialize_components()
        
        # Document metadata and state
        self.is_document_processed = False
    
    def _initialize_components(self):
        """
        Initialize the workflow components.
        """
        # Import here to avoid circular imports
        logger.info("Initializing QA workflow")
        
        # Import necessary components
        from pdf_loader import PDFProcessor
        from utils.knowledge_graph import DocumentKnowledgeGraph
        
        # Initialize PDF processor with document_path
        self.pdf_processor = PDFProcessor(
            document_path=self.document_path,
            heading_split=True
        )
        
        # Initialize other components
        self._initialize_embeddings()
        
        if self.use_knowledge_graph:
            # Initialize fallback knowledge graph directly
            try:
                self.knowledge_graph = DocumentKnowledgeGraph(
                    space_name=os.path.basename(self.document_path),
                    persist_dir=self.persist_dir,
                    llm=Settings.llm
                )
            except Exception as e:
                logger.error(f"Error initializing knowledge graph: {e}")
                self.knowledge_graph = None

    def _initialize_embeddings(self):
        """
        Initialize embeddings.
        """
        try:
            from utils.embeddings import DocumentEmbeddings
            
            # Initialize document embeddings
            self.embeddings = DocumentEmbeddings(
                persist_dir=self.persist_dir,
                openai_api_key=self.openai_api_key  # Pass API key
            )
            
            # For backward compatibility, also set embeddings_manager
            self.embeddings_manager = self.embeddings
            
        except ImportError as e:
            logger.error(f"Error initializing embeddings: {e}")
            logger.warning("Continuing without embeddings support")
            self.embeddings = None
            self.embeddings_manager = None
        except Exception as e:
            logger.error(f"Unexpected error initializing embeddings: {e}")
            logger.warning("Continuing without embeddings support")
            self.embeddings = None
            self.embeddings_manager = None
        
        # Initialize agents
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent(
            embeddings_manager=self.embeddings_manager,
            knowledge_graph=self.knowledge_graph
        )
        self.answer_generator = AnswerGeneratorAgent()
    
    def process_document(self) -> Dict[str, Any]:
        """
        Process the document and prepare it for question answering.
        
        Returns:
            Document metadata
        """
        if self.is_document_processed:
            return self.document_metadata
        
        logger.info(f"Processing document: {self.document_path}")
        
        try:
            # Try the process_document method first
            try:
                self.document_chunks, self.document_metadata = self.pdf_processor.process_document()
            except AttributeError:
                # Fall back to the process method
                result = self.pdf_processor.process()
                self.document_chunks = result["chunks"]
                self.document_metadata = result["metadata"]
                self.document_text = result.get("text", "")
            
            logger.info(f"Document processed: {len(self.document_chunks)} chunks")
            
            # Create a document dictionary for processing
            document_dict = {
                "chunks": self.document_chunks,
                "metadata": self.document_metadata,
                "text": self.document_text
            }
            
            # Process document with embeddings - with safe attribute check
            if hasattr(self, 'embeddings') and self.embeddings:
                try:
                    logger.info("Indexing document with embeddings...")
                    self.embeddings.process_document(document_dict)
                    
                    # Explicitly call index_documents if it exists
                    if hasattr(self.embeddings, 'index_documents'):
                        logger.info("Calling index_documents on embeddings...")
                        self.embeddings.index_documents(self.document_chunks, 
                                                        metadata=[self.document_metadata] * len(self.document_chunks))
                except Exception as e:
                    logger.error(f"Error processing document with embeddings: {e}")
                    logger.warning("Continuing without embeddings")
            
            # Process document with knowledge graph - with safe attribute check
            if self.use_knowledge_graph and hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                try:
                    logger.info("Processing document with knowledge graph...")
                    self.knowledge_graph.process_document(document_dict)
                except Exception as e:
                    logger.error(f"Error processing document with knowledge graph: {e}")
                    logger.warning("Continuing without knowledge graph")
            
            # Store document metadata in conversation memory
            self.memory.set_document_metadata(self.document_metadata)
            
            # Mark document as processed
            self.is_document_processed = True
            
            return self.document_metadata
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question and generate an answer.
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing answer and related information
        """
        # Ensure document is processed
        if not self.is_document_processed:
            self.process_document()
        
        try:
            # Step 1: Plan query processing
            logger.info(f"Planning query: {question}")
            query_analysis, search_strategy = self.planner.plan_retrieval(
                question=question,
                document_metadata=self.document_metadata,
                conversation_history=self.memory.get_conversation_history()
            )
            
            # Step 2: Retrieve relevant context
            logger.info("Retrieving context")
            retrieved_nodes = self.retriever.retrieve_context(
                query=question,
                search_strategy=search_strategy,
                document_metadata=self.document_metadata
            )
            
            # Step 3: Generate answer
            logger.info("Generating answer")
            answer = self.answer_generator.generate_answer(
                question=question,
                retrieved_nodes=retrieved_nodes,
                query_analysis=query_analysis,
                document_metadata=self.document_metadata,
                conversation_memory=self.memory
            )
            
            # Step 4: Sanitize any potential hallucinations
            context_text = "\n".join([node.text for node in retrieved_nodes])
            sanitized_answer = self.answer_generator.sanitize_hallucinations(
                answer=answer,
                context=context_text
            )
            
            # Step 5: Store interaction in memory
            self.memory.add_interaction(
                question=question,
                answer=sanitized_answer,
                context_docs=[{
                    "id": node.node_id if hasattr(node, 'node_id') else str(i),
                    "text": node.text,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                } for i, node in enumerate(retrieved_nodes)],
                metadata={
                    "query_type": query_analysis.get("query_type"),
                    "timestamp": None  # Will be added by memory module
                }
            )
            
            # Return comprehensive result
            result = {
                "question": question,
                "answer": sanitized_answer,
                "query_analysis": query_analysis,
                "search_strategy": search_strategy,
                "context_count": len(retrieved_nodes),
                "is_follow_up": query_analysis.get("is_follow_up", False)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            error_result = {
                "question": question,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "error": str(e)
            }
            return error_result
"""
Retriever agent for context fetching from document store.
"""
from typing import Dict, Any, List, Optional

from llama_index.core.schema import Document, NodeWithScore

class RetrieverAgent:
    """
    Retriever agent responsible for fetching relevant context based on search strategy.
    Uses vector search, hybrid search, and filtering to retrieve document chunks
    relevant to a query based on the strategy from the planner.
    """
    
    def __init__(
        self,
        embeddings_manager,
        knowledge_graph=None
    ):
        """
        Initialize the retriever agent.
        
        Args:
            embeddings_manager: Manager for vector embeddings
            knowledge_graph: Optional knowledge graph manager
        """
        self.embeddings_manager = embeddings_manager
        self.knowledge_graph = knowledge_graph
    
    def retrieve_context(
        self,
        query: str,
        search_strategy: Dict[str, Any],
        document_metadata: Dict[str, Any]
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant context based on search strategy.
        
        Args:
            query: The user's query
            search_strategy: Search strategy from planner
            document_metadata: Document metadata
            
        Returns:
            List of retrieved document chunks with relevance scores
        """
        # Extract search parameters
        query_type = search_strategy.get("query_type", "unknown")
        vector_search_terms = search_strategy.get("vector_search_terms", [])
        keyword_filters = search_strategy.get("keyword_filters", [])
        hybrid_weight = search_strategy.get("hybrid_search_weight", 0.7)
        top_k = search_strategy.get("top_k", 5)
        include_tables = search_strategy.get("relevant_tables", False)
        include_charts = search_strategy.get("relevant_charts", False)
        
        # If vector search terms provided, use them to enhance the query
        enhanced_query = query
        if vector_search_terms:
            enhanced_query = f"{query} {' '.join(vector_search_terms)}"
        
        # Perform base retrieval using embeddings_manager semantic search
        if hasattr(self.embeddings_manager, 'get_relevant_chunks'):
            chunk_texts = self.embeddings_manager.get_relevant_chunks(
                enhanced_query,
                top_k=top_k * 2
            )
            # Wrap chunks in Document nodes
            retrieved_nodes = [Document(text=ct) for ct in chunk_texts]
        else:
            retrieved_nodes = []
        
        # Apply keyword filters if specified
        if keyword_filters:
            filtered_nodes = []
            for node in retrieved_nodes:
                text = node.text.lower()
                # Check if any keyword is in the text
                if any(keyword.lower() in text for keyword in keyword_filters):
                    filtered_nodes.append(node)
              # If no nodes match the filter, fall back to the original nodes
            retrieved_nodes = filtered_nodes if filtered_nodes else retrieved_nodes
        
        # Include tables if requested and available
        if include_tables and "tables" in document_metadata:
            table_nodes = self._retrieve_relevant_tables(query, document_metadata["tables"])
            retrieved_nodes.extend(table_nodes)
        
        # Include charts/figures if requested and available
        if include_charts and "images" in document_metadata:
            chart_nodes = self._retrieve_relevant_charts(query, document_metadata["images"])
            retrieved_nodes.extend(chart_nodes)
        
        # Use knowledge graph for enhanced retrieval if available and beneficial
        if self.knowledge_graph and self.knowledge_graph.available and query_type in ["analytical", "comparative"]:
            kg_results = self._retrieve_from_knowledge_graph(query)
            if kg_results:
                # Convert KG results to nodes and add them
                kg_nodes = self._convert_kg_to_nodes(kg_results)
                retrieved_nodes.extend(kg_nodes)
        
        # Sort by relevance score and limit to top_k
        retrieved_nodes.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        retrieved_nodes = retrieved_nodes[:top_k]
        
        return retrieved_nodes
    
    def _retrieve_relevant_tables(
        self,
        query: str,
        tables: List[Dict[str, Any]]
    ) -> List[NodeWithScore]:
        """
        Retrieve tables relevant to the query.
        
        Args:
            query: The user's query
            tables: List of tables from document metadata
            
        Returns:
            List of nodes representing relevant tables
        """
        table_nodes = []
        
        # Simplified relevance scoring based on keyword matching
        # Could be enhanced with more sophisticated methods
        for table in tables:
            # Convert table text representation to a document
            table_text = table.get("text_representation", "")
            if not table_text:
                continue
                
            # Simple relevance scoring - count query term occurrences
            score = 0
            for term in query.lower().split():
                if len(term) > 3:  # Ignore short terms
                    score += table_text.lower().count(term)
            
            # Normalize score
            score = min(score / 10, 1.0)
            
            # If score is above threshold, add to results
            if score > 0.2:
                # Create node for the table
                node = Document(
                    text=f"Table from page {table.get('page', 'unknown')}:\n{table_text}",
                    metadata={
                        "source": "table",
                        "page": table.get("page"),
                        "id": table.get("id")
                    }
                )
                node.score = score
                table_nodes.append(node)
        
        return table_nodes
    
    def _retrieve_relevant_charts(
        self,
        query: str,
        images: List[Dict[str, Any]]
    ) -> List[NodeWithScore]:
        """
        Retrieve charts/figures relevant to the query.
        
        Args:
            query: The user's query
            images: List of images from document metadata
            
        Returns:
            List of nodes representing relevant charts
        """
        chart_nodes = []
        
        # Filter for images that are charts
        charts = [img for img in images if img.get("is_chart", False)]
        
        # Score charts based on caption relevance
        for chart in charts:
            caption = chart.get("caption", "")
            if not caption:
                continue
                
            # Simple relevance scoring - count query term occurrences
            score = 0
            for term in query.lower().split():
                if len(term) > 3:  # Ignore short terms
                    score += caption.lower().count(term)
            
            # Normalize score
            score = min(score / 5, 1.0)
            
            # If score is above threshold, add to results
            if score > 0.2:
                # Create node for the chart
                node = Document(
                    text=f"Chart/Figure from page {chart.get('page', 'unknown')}:\n{caption}",
                    metadata={
                        "source": "chart",
                        "page": chart.get("page"),
                        "id": chart.get("id")
                    }
                )
                node.score = score
                chart_nodes.append(node)
        
        return chart_nodes
    
    def _retrieve_from_knowledge_graph(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve information from knowledge graph.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with knowledge graph results, or None if not available
        """
        if not self.knowledge_graph:
            return None
            
        try:
            # Query the knowledge graph
            kg_results = self.knowledge_graph.query_graph(query)
            return kg_results
        except Exception as e:
            print(f"Error querying knowledge graph: {e}")
            return None
    
    def _convert_kg_to_nodes(self, kg_results: Dict[str, Any]) -> List[NodeWithScore]:
        """
        Convert knowledge graph results to document nodes.
        
        Args:
            kg_results: Results from knowledge graph query
            
        Returns:
            List of nodes derived from knowledge graph
        """
        kg_nodes = []
        
        # Extract source nodes from KG results if available
        source_nodes = kg_results.get("source_nodes", [])
        for node in source_nodes:
            # Convert to document node
            doc_node = Document(
                text=node.get("text", ""),
                metadata={
                    "source": "knowledge_graph",
                    "node_id": node.get("id", ""),
                    **node.get("metadata", {})
                }
            )
            doc_node.score = node.get("score", 0.5)
            kg_nodes.append(doc_node)
        
        return kg_nodes
"""
Planner agent for query analysis and retrieval strategy.
"""
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from llama_index.core.prompts import PromptTemplate

class QueryType(Enum):
    """
    Enum for different types of queries.
    """
    FACTUAL = "factual"          # Direct fact retrieval
    SUMMARY = "summary"          # Summary of document section
    ANALYTICAL = "analytical"    # Analysis requiring reasoning
    COMPARATIVE = "comparative"  # Comparison between different sections
    FOLLOW_UP = "follow_up"      # Follow-up to previous question
    UNKNOWN = "unknown"          # Unable to categorize

class PlannerAgent:
    """
    Planner agent that analyzes queries and determines the retrieval strategy.
    Responsible for understanding the query intent and planning the best approach
    to find relevant information in the document.
    """
    
    def __init__(self, temperature: float = 0.0):
        """
        Initialize the planner agent.
        
        Args:
            temperature: Temperature for LLM generation
        """
        self.llm = OpenAI(temperature=temperature, model="gpt-4o")
        
        # Define prompt templates
        self._define_prompts()
    
    def _define_prompts(self):
        """
        Define the prompt templates for the planner.
        """
        # Query analysis prompt
        self.query_analysis_template = PromptTemplate(
            template="""
            You are a planning agent for a document question answering system.
            Your task is to analyze the user's question and determine:
            1. The type of question being asked
            2. The key information needed to answer it
            3. Where in the document this information might be found
            4. How many relevant sections would be needed
            
            Here is information about the document:
            - Title: {document_title}
            - Topics: {document_topics}
            - Sections: {document_sections}
            
            User Question: {question}
            
            Previous conversation history:
            {conversation_history}
            
            Analyze the question and provide the following information in JSON format:
            - query_type: One of ["factual", "summary", "analytical", "comparative", "follow_up", "unknown"]
            - key_terms: List of key terms to search for
            - likely_sections: List of document sections where the answer might be found
            - required_context_amount: How many chunks needed (few, moderate, extensive)
            - is_follow_up: Whether this appears to be a follow-up question to previous conversation
            - related_entities: Entities (people, organizations, concepts) mentioned or implied
            
            JSON Response:
            """
        )
        
        # Search strategy prompt
        self.search_strategy_template = PromptTemplate(
            template="""
            Based on the query analysis, determine the best search strategy.
            
            Query Analysis: {query_analysis}
            
            Search Strategy Response:
            - vector_search_terms: List of terms for semantic search
            - keyword_filters: Optional keywords to filter results
            - relevant_tables: Whether tables are likely needed (true/false)
            - relevant_charts: Whether charts/figures are likely needed (true/false)
            - hybrid_search_weight: Relative importance of semantic vs keyword search (0.0-1.0)
            """
        )
    
    def analyze_query(
        self,
        question: str,
        document_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and information needs.
        
        Args:
            question: The user's question
            document_metadata: Metadata about the document
            conversation_history: Previous conversation history (optional)
            
        Returns:
            Dictionary with query analysis
        """
        # Extract document information
        document_title = document_metadata.get("title", "Unknown Document")
        document_topics = document_metadata.get("keywords", "")
        
        # Get document sections from TOC if available
        document_sections = []
        for item in document_metadata.get("toc", []):
            section = f"{'-' * item['level']} {item['title']} (Page {item['page']})"
            document_sections.append(section)
        
        # Format document sections
        formatted_sections = "\n".join(document_sections) if document_sections else "No section information available"
        
        # Format conversation history
        formatted_history = ""
        if conversation_history and len(conversation_history) > 0:
            for i, turn in enumerate(conversation_history[-3:]):  # Last 3 turns
                formatted_history += f"Q{i+1}: {turn['question']}\n"
                formatted_history += f"A{i+1}: {turn['answer'][:200]}...\n\n"
        else:
            formatted_history = "No previous conversation"
        
        # Prepare the query analysis prompt
        query_analysis_prompt = self.query_analysis_template.format(
            document_title=document_title,
            document_topics=document_topics,
            document_sections=formatted_sections,
            question=question,
            conversation_history=formatted_history
        )
        
        # Get response from LLM
        query_analysis_response = self.llm.complete(query_analysis_prompt)
        
        # Parse the JSON response
        try:
            import json
            query_analysis = json.loads(query_analysis_response.text)
        except json.JSONDecodeError:
            # Fallback if response is not valid JSON
            query_analysis = {
                "query_type": "unknown",
                "key_terms": [],
                "likely_sections": [],
                "required_context_amount": "moderate",
                "is_follow_up": False,
                "related_entities": []
            }
        
        # Add original question to the analysis
        query_analysis["question"] = question
        
        return query_analysis
    
    def determine_search_strategy(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the search strategy based on query analysis.
        
        Args:
            query_analysis: The query analysis from analyze_query
            
        Returns:
            Dictionary with search strategy
        """
        # Prepare the search strategy prompt
        search_strategy_prompt = self.search_strategy_template.format(
            query_analysis=str(query_analysis)
        )
        
        # Get response from LLM
        search_strategy_response = self.llm.complete(search_strategy_prompt)
        
        # Parse the response (assuming it's in a structured format)
        strategy = self._parse_search_strategy(search_strategy_response.text)
        
        # Add the query type from analysis
        strategy["query_type"] = query_analysis.get("query_type", "unknown")
        
        # Determine number of results to retrieve based on context amount
        context_amount = query_analysis.get("required_context_amount", "moderate")
        if context_amount == "few":
            strategy["top_k"] = 3
        elif context_amount == "moderate":
            strategy["top_k"] = 5
        elif context_amount == "extensive":
            strategy["top_k"] = 8
        else:
            strategy["top_k"] = 5  # Default
        
        return strategy
    
    def _parse_search_strategy(self, strategy_text: str) -> Dict[str, Any]:
        """
        Parse the search strategy response.
        
        Args:
            strategy_text: The raw response from LLM
            
        Returns:
            Dictionary with parsed strategy
        """
        # Default values
        strategy = {
            "vector_search_terms": [],
            "keyword_filters": [],
            "relevant_tables": False,
            "relevant_charts": False,
            "hybrid_search_weight": 0.7  # Default to 70% semantic, 30% keyword
        }
        
        # Parse each line of the response
        for line in strategy_text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
                
            key, value = line.split(":", 1)
            key = key.strip().replace("-", "").strip()
            value = value.strip()
            
            if key == "vector_search_terms":
                # Extract list of terms
                terms = [t.strip() for t in value.strip("[]").split(",")]
                strategy["vector_search_terms"] = [t for t in terms if t]
            elif key == "keyword_filters":
                # Extract list of keywords
                keywords = [k.strip() for k in value.strip("[]").split(",")]
                strategy["keyword_filters"] = [k for k in keywords if k]
            elif key == "relevant_tables":
                strategy["relevant_tables"] = value.lower() == "true"
            elif key == "relevant_charts":
                strategy["relevant_charts"] = value.lower() == "true"
            elif key == "hybrid_search_weight":
                try:
                    weight = float(value)
                    strategy["hybrid_search_weight"] = max(0.0, min(1.0, weight))
                except ValueError:
                    pass  # Keep default
        
        return strategy
    
    def plan_retrieval(
        self,
        question: str,
        document_metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Plan the retrieval strategy for a query.
        
        Args:
            question: The user's question
            document_metadata: Metadata about the document
            conversation_history: Previous conversation history (optional)
            
        Returns:
            Tuple containing:
                - Query analysis
                - Search strategy
        """
        # First analyze the query
        query_analysis = self.analyze_query(
            question=question,
            document_metadata=document_metadata,
            conversation_history=conversation_history
        )
        
        # Then determine search strategy
        search_strategy = self.determine_search_strategy(query_analysis)
        
        return query_analysis, search_strategy
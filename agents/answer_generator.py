"""
Answer generator agent that composes coherent, contextual responses.
"""
from typing import Dict, Any, List, Optional

from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.prompts import PromptTemplate

class AnswerGeneratorAgent:
    """
    Answer generation agent that produces coherent responses based on retrieved context.
    Uses retrieved document chunks and query information to construct
    comprehensive and accurate answers to user questions.
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize the answer generator agent.
        
        Args:
            temperature: Temperature for LLM generation
        """
        self.llm = OpenAI(temperature=temperature, model="gpt-4o")
        
        # Define prompt templates
        self._define_prompts()
    
    def _define_prompts(self):
        """
        Define the prompt templates for answer generation.
        """
        # Standard answer generation prompt
        self.answer_generation_template = PromptTemplate(
            template="""
            You are an assistant tasked with answering questions about business documents.
            Your role is to provide accurate, concise, and informative responses based solely on the provided context.
            
            Document Title: {document_title}
            Question: {question}
            
            Relevant Context:
            {context}
            
            Instructions:
            1. Answer the question using ONLY the information in the provided context.
            2. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
            3. If you need to make assumptions, clearly state them.
            4. Do NOT introduce information from your general knowledge that isn't present in the context.
            5. Organize the answer with clear structure when appropriate (bullet points, numbering, etc.)
            6. Include specific details from the context to support your answer.
            7. If there are direct quotes or statistics, include page references in parentheses where available.
            
            Provide a comprehensive, well-structured answer:
            """
        )
        
        # Follow-up question prompt template
        self.follow_up_template = PromptTemplate(
            template="""
            You are answering a follow-up question in a conversation about a business document.
            
            Document Title: {document_title}
            Previous Question: {previous_question}
            Previous Answer: {previous_answer}
            Follow-up Question: {question}
            
            Relevant Context:
            {context}
            
            Instructions:
            1. Consider both the new context and the previous Q&A when formulating your response.
            2. Directly address the follow-up question while maintaining continuity with the previous answer.
            3. If the follow-up question relies on information from the previous answer, incorporate that.
            4. Answer the question using ONLY the information in the provided context and previous Q&A.
            5. If the context doesn't contain enough information, acknowledge this limitation.
            6. Organize the answer with clear structure when appropriate.
            
            Provide a comprehensive, well-structured answer:
            """
        )
        
        # Analytical question prompt template
        self.analytical_template = PromptTemplate(
            template="""
            You are answering an analytical question about a business document that requires reasoning and insight.
            
            Document Title: {document_title}
            Question: {question}
            Query Type: Analytical
            
            Relevant Context:
            {context}
            
            Instructions:
            1. Analyze the information in the context to provide meaningful insights.
            2. Identify patterns, implications, or connections between different pieces of information.
            3. Structure your response with a clear analysis framework:
               - Key observations
               - Analysis of implications
               - Reasoned conclusions
            4. Support your analysis with specific evidence from the context.
            5. Be balanced and objective in your assessment.
            6. Acknowledge limitations in the available information when relevant.
            
            Provide a thoughtful, well-reasoned analytical response:
            """
        )
        
        # Summary question prompt template
        self.summary_template = PromptTemplate(
            template="""
            You are summarizing information from a business document.
            
            Document Title: {document_title}
            Request: {question}
            Query Type: Summary
            
            Content to Summarize:
            {context}
            
            Instructions:
            1. Provide a concise, comprehensive summary of the key points in the content.
            2. Maintain the essential information, main arguments, and important details.
            3. Structure the summary in a logical flow that captures the original content's organization.
            4. Use clear sections or bullet points for readability where appropriate.
            5. Include any critical figures, statistics, or quotes where relevant.
            6. Ensure the summary stands on its own as a complete overview.
            
            Provide a well-structured, informative summary:
            """
        )
    
    def generate_answer(
        self,
        question: str,
        retrieved_nodes: List[NodeWithScore],
        query_analysis: Dict[str, Any],
        document_metadata: Dict[str, Any],
        conversation_memory=None
    ) -> str:
        """
        Generate an answer based on retrieved context.
        
        Args:
            question: The user's question
            retrieved_nodes: List of retrieved document chunks
            query_analysis: Analysis of the query from planner
            document_metadata: Document metadata
            conversation_memory: Optional conversation memory
            
        Returns:
            Generated answer
        """
        # Extract query type
        query_type = query_analysis.get("query_type", "unknown")
        
        # Get document title
        document_title = document_metadata.get("title", "Unknown Document")
        
        # Format context from retrieved nodes
        formatted_context = self._format_context(retrieved_nodes)
        
        # Check if this is a follow-up question
        is_follow_up = query_analysis.get("is_follow_up", False)
        
        # Select appropriate template based on query type
        if is_follow_up and conversation_memory:
            # Get previous Q&A
            history = conversation_memory.get_conversation_history(last_n=1)
            if history:
                previous_qa = history[0]
                previous_question = previous_qa.get("question", "")
                previous_answer = previous_qa.get("answer", "")
                
                # Use follow-up template
                prompt = self.follow_up_template.format(
                    document_title=document_title,
                    previous_question=previous_question,
                    previous_answer=previous_answer,
                    question=question,
                    context=formatted_context
                )
            else:
                # Fall back to standard template
                prompt = self.answer_generation_template.format(
                    document_title=document_title,
                    question=question,
                    context=formatted_context
                )
        elif query_type == "analytical":
            # Use analytical template
            prompt = self.analytical_template.format(
                document_title=document_title,
                question=question,
                context=formatted_context
            )
        elif query_type == "summary":
            # Use summary template
            prompt = self.summary_template.format(
                document_title=document_title,
                question=question,
                context=formatted_context
            )
        else:
            # Use standard template
            prompt = self.answer_generation_template.format(
                document_title=document_title,
                question=question,
                context=formatted_context
            )
        
        # Generate the answer
        response = self.llm.complete(prompt)
        
        # Check for empty or very short responses
        if not response.text or len(response.text.strip()) < 10:
            return "I don't have enough information to answer this question based on the document content."
        
        return response.text.strip()
    
    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved nodes into context string.
        
        Args:
            nodes: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        # Sort nodes by score if available
        sorted_nodes = sorted(
            nodes, 
            key=lambda x: x.score if hasattr(x, 'score') and x.score is not None else 0,
            reverse=True
        )
        
        formatted_chunks = []
        for i, node in enumerate(sorted_nodes):
            # Extract text and metadata
            text = node.text if hasattr(node, 'text') else str(node)
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            # Get source information
            source_type = metadata.get("source", "document")
            page_num = metadata.get("page", "unknown")
            
            # Format based on source type
            if source_type == "table":
                formatted_chunk = f"TABLE (Page {page_num}):\n{text}\n"
            elif source_type == "chart":
                formatted_chunk = f"CHART (Page {page_num}):\n{text}\n"
            elif source_type == "knowledge_graph":
                formatted_chunk = f"RELATED CONCEPT:\n{text}\n"
            else:
                formatted_chunk = f"EXCERPT {i+1} (Page {page_num}):\n{text}\n"
            
            formatted_chunks.append(formatted_chunk)
        
        # Join all formatted chunks
        return "\n".join(formatted_chunks)
    
    def sanitize_hallucinations(self, answer: str, context: str) -> str:
        """
        Detect and sanitize potential hallucinations in the generated answer.
        
        Args:
            answer: The generated answer
            context: The context used to generate the answer
            
        Returns:
            Sanitized answer with hallucination warnings if detected
        """
        # This is a simplified implementation
        # A more sophisticated approach would involve entity extraction and verification
        
        # Check for statements that make strong assertions not supported by context
        hallucination_markers = [
            "clearly shows",
            "definitively states",
            "explicitly mentions",
            "proves that",
            "confirms that",
            "demonstrates unequivocally",
            "100%",
            "absolute"
        ]
        
        # Flag potentially hallucinated statements
        sanitized_answer = answer
        for marker in hallucination_markers:
            if marker in answer.lower():
                # Check if this assertion is supported in the context
                sentence_with_marker = next((s for s in answer.split('.') if marker in s.lower()), "")
                if sentence_with_marker and not self._is_supported_by_context(sentence_with_marker, context):
                    # Replace strong assertion with a more cautious statement
                    replacement = sentence_with_marker.replace(marker, "suggests").strip()
                    sanitized_answer = sanitized_answer.replace(sentence_with_marker, f"{replacement} (though this may be an interpretation)")
        
        return sanitized_answer
    
    def _is_supported_by_context(self, statement: str, context: str) -> bool:
        """
        Check if a statement is supported by the context.
        
        Args:
            statement: The statement to check
            context: The context
            
        Returns:
            Boolean indicating if statement is supported
        """
        # Simplified implementation - check for key terms from statement in context
        # A more sophisticated approach would use entailment detection
        
        # Get key terms (excluding common words)
        common_words = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "that", "this", "is", "are", "was", "were"}
        statement_terms = [
            term.lower() for term in statement.split() 
            if term.lower() not in common_words and len(term) > 3
        ]
        
        # Check if most key terms are in the context
        context_lower = context.lower()
        matches = sum(1 for term in statement_terms if term in context_lower)
        match_ratio = matches / len(statement_terms) if statement_terms else 0
        
        return match_ratio >= 0.7  # At least 70% of terms should be present
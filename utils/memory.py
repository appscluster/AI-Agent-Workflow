"""
Conversation memory for tracking context and history in multi-turn interactions.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ConversationMemory:
    """
    Manages conversation history and context for multi-turn interactions.
    Maintains a record of previous questions, answers, and relevant context
    to enable contextual follow-up questions.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of turns to remember
        """
        self.max_history = max_history
        self.history = []
        self.current_context = {}
        self.document_metadata = {}
    
    def add_interaction(
        self, 
        question: str, 
        answer: str, 
        context_docs: Optional[List[Dict[str, Any]]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a new interaction to memory.
        
        Args:
            question: User's question
            answer: System's answer
            context_docs: List of context documents used
            metadata: Additional metadata about the interaction
        """
        timestamp = datetime.now().isoformat()
        
        # Create interaction record
        interaction = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "context_references": self._extract_context_references(context_docs) if context_docs else [],
            "metadata": metadata or {}
        }
        
        # Add to history
        self.history.append(interaction)
        
        # Trim history if exceeds max length
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def _extract_context_references(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract minimal references from context documents.
        
        Args:
            context_docs: Full context documents
            
        Returns:
            List of minimal context references
        """
        references = []
        for doc in context_docs:
            # Extract just enough to reference this later
            ref = {
                "id": doc.get("id", str(hash(doc.get("text", "")))),
                "text_snippet": doc.get("text", "")[:100] + "..." if len(doc.get("text", "")) > 100 else doc.get("text", ""),
                "metadata": {
                    k: v for k, v in doc.get("metadata", {}).items() 
                    if k in ["page", "source", "title", "section"]
                }
            }
            references.append(ref)
        
        return references
    
    def set_document_metadata(self, metadata: Dict[str, Any]):
        """
        Set metadata about the current document being processed.
        
        Args:
            metadata: Document metadata
        """
        self.document_metadata = metadata
    
    def update_current_context(self, context: Dict[str, Any]):
        """
        Update the current context dict with new information.
        
        Args:
            context: New context information
        """
        self.current_context.update(context)
    
    def clear_current_context(self):
        """
        Clear the current context.
        """
        self.current_context = {}
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            last_n: Number of most recent interactions to return (optional)
            
        Returns:
            List of interaction records
        """
        if last_n is None or last_n >= len(self.history):
            return self.history
        
        return self.history[-last_n:]
    
    def get_recent_questions(self, n: int = 3) -> List[str]:
        """
        Get the most recent user questions.
        
        Args:
            n: Number of recent questions to return
            
        Returns:
            List of recent questions
        """
        return [turn["question"] for turn in self.history[-n:]]
    
    def get_follow_up_context(self) -> Dict[str, Any]:
        """
        Get context specifically for follow-up questions.
        Combines current context with relevant history.
        
        Returns:
            Dictionary with context for follow-up questions
        """
        # Start with current context
        follow_up_context = dict(self.current_context)
        
        # Add recent interaction summary
        if self.history:
            recent = self.history[-1]
            follow_up_context["previous_question"] = recent["question"]
            follow_up_context["previous_answer"] = recent["answer"]
            
            # Add reference to context used in previous answer
            if recent.get("context_references"):
                follow_up_context["previous_context_references"] = recent["context_references"]
        
        # Add document metadata
        if self.document_metadata:
            follow_up_context["document_metadata"] = self.document_metadata
        
        return follow_up_context
    
    def save_to_file(self, filepath: str):
        """
        Save conversation history to a file.
        
        Args:
            filepath: Path to save the file
        """
        data = {
            "history": self.history,
            "document_metadata": self.document_metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Load conversation history from a file.
        
        Args:
            filepath: Path to the file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.history = data.get("history", [])
        self.document_metadata = data.get("document_metadata", {})
    
    def detect_follow_up_question(self, question: str) -> bool:
        """
        Detect if a question is likely a follow-up to previous conversation.
        
        Args:
            question: The current question
            
        Returns:
            Boolean indicating if question is likely a follow-up
        """
        # No history means it can't be a follow-up
        if not self.history:
            return False
        
        # Check for pronouns and other follow-up indicators
        follow_up_indicators = [
            "it", "this", "that", "they", "these", "those", 
            "their", "its", "the", "they", "them",
            "what about", "how about", "what else", "tell me more",
            "why", "how", "when", "where", "who", "which"
        ]
        
        # Convert to lowercase for comparison
        question_lower = question.lower()
        
        # Check if question starts with a follow-up indicator
        for indicator in follow_up_indicators:
            if question_lower.startswith(indicator + " ") or question_lower == indicator:
                return True
        
        # Check if question contains references to previous context
        if "previous" in question_lower or "earlier" in question_lower or "before" in question_lower:
            return True
        
        # If question is very short, it's likely a follow-up
        if len(question.split()) <= 3:
            return True
        
        return False
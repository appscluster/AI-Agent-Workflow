"""
LLM client for generating answers.
"""
import os
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Language Model APIs.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Some functionality may be limited.")
    
    def generate_answer(self, question: str, contexts: List[str], max_tokens: int = 2000) -> str:
        """
        Generate answer for a question using provided contexts.
        
        Args:
            question: Question to answer
            contexts: List of context texts
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated answer
        """
        try:
            import openai
            
            # Set API key
            openai.api_key = self.openai_api_key
            
            # Prepare context
            context_text = "\n\n".join(contexts)
            
            # Truncate context if too long (rough estimate)
            if len(context_text) > 15000:
                logger.warning("Context is too long, truncating...")
                context_text = context_text[:15000] + "..."
            
            # Prepare prompt
            prompt = f"""
            I need you to answer a question based on the following context:
            
            CONTEXT:
            {context_text}
            
            QUESTION:
            {question}
            
            Please provide a comprehensive and accurate answer based only on the information in the context.
            If the context doesn't contain enough information to answer the question, say so.
            """
            
            # Generate response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            return answer
        
        except ImportError:
            logger.error("OpenAI package not installed")
            return "I'm sorry, but I can't generate an answer because the OpenAI package is not installed."
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I'm sorry, but I encountered an error while generating an answer: {str(e)}"

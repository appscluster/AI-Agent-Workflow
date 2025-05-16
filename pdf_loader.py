"""
PDF loading and processing utilities.
"""
import os
import logging
import re
from typing import List, Dict, Any, Optional
import PyPDF2
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Class for loading and processing PDF documents.
    """
    
    def __init__(
        self, 
        document_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        heading_split: bool = False  # Added heading_split parameter
    ):
        """
        Initialize PDF processor.
        
        Args:
            document_path: Path to PDF document
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            heading_split: Whether to split text at headings
        """
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_split = heading_split  # Store the heading_split flag
        
        # Validate file exists
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Validate file is PDF
        if not document_path.lower().endswith('.pdf'):
            raise ValueError(f"Document must be a PDF file: {document_path}")
    
    def extract_text(self) -> str:
        """
        Extract text from PDF document.
        
        Returns:
            Extracted text from the PDF
        """
        try:
            logger.info(f"Extracting text from {self.document_path}")
            text = extract_text(self.document_path)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.
        
        Returns:
            Dictionary of metadata
        """
        try:
            logger.info(f"Extracting metadata from {self.document_path}")
            
            metadata = {}
            with open(self.document_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic metadata
                metadata['pages'] = len(reader.pages)
                
                # PDF info dictionary
                if reader.metadata:
                    for key, value in reader.metadata.items():
                        if key.startswith('/'):
                            key = key[1:]  # Remove leading slash
                        metadata[key] = value
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from PDF: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # If heading_split is enabled, split at headings first
        if self.heading_split:
            return self._chunk_text_by_headings(text)
        else:
            return self._chunk_text_by_size(text)
    
    def _chunk_text_by_size(self, text: str) -> List[str]:
        """
        Split text into chunks based on size.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk with specified size
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end and not at a whitespace, move back to find whitespace
            if end < len(text) and not text[end].isspace():
                # Find previous whitespace
                while end > start and not text[end].isspace():
                    end -= 1
                
                # If no whitespace found in chunk, use chunk_size
                if end == start:
                    end = min(start + self.chunk_size, len(text))
            
            # Add chunk to list
            chunks.append(text[start:end])
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end - self.chunk_overlap > start else end
        
        return chunks
    
    def _chunk_text_by_headings(self, text: str) -> List[str]:
        """
        Split text into chunks based on headings.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Regular expression for common heading patterns
        heading_patterns = [
            r'\n\s*#+\s+.*?\n',  # Markdown headings
            r'\n\s*[A-Z][A-Za-z\s]+:\s*\n',  # Title with colon
            r'\n\s*[IVXLCDM]+\.\s+.*?\n',  # Roman numerals
            r'\n\s*[A-Z][A-Za-z\s]+(:|\.)\s*\n',  # Capitalized headings
            r'\n\s*\d+\.\d*\s+.*?\n',  # Numbered headings (1.1, 2.3, etc.)
            r'\n\s*\d+\.\s+.*?\n',  # Numbered headings (1., 2., etc.)
        ]
        
        # Combine patterns
        pattern = '|'.join(heading_patterns)
        
        # Find all headings
        headings = re.finditer(pattern, text)
        heading_positions = [0]  # Start with beginning of text
        
        for match in headings:
            heading_positions.append(match.start())
        
        heading_positions.append(len(text))  # End with end of text
        
        chunks = []
        
        # For each section between headings
        for i in range(len(heading_positions) - 1):
            section_start = heading_positions[i]
            section_end = heading_positions[i + 1]
            section_text = text[section_start:section_end]
            
            # If section is too large, split further by size
            if len(section_text) > self.chunk_size:
                section_chunks = self._chunk_text_by_size(section_text)
                chunks.extend(section_chunks)
            else:
                chunks.append(section_text)
        
        return chunks
    
    def process_document(self) -> tuple:
        """
        Process document and return chunks and metadata.
        This method is for compatibility with the workflow graph.
        
        Returns:
            Tuple of (chunks, metadata)
        """
        result = self.process()
        return result["chunks"], result["metadata"]
    
    def process(self) -> Dict[str, Any]:
        """
        Process document and extract text and metadata.
        
        Returns:
            Dictionary with text and metadata
        """
        # Extract text
        text = self.extract_text()
        
        # Extract metadata
        metadata = self.extract_metadata()
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        # Return processed document
        return {
            "text": text,
            "metadata": metadata,
            "chunks": chunks
        }

def load_pdf(document_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """
    Load and process PDF document.
    
    Args:
        document_path: Path to PDF document
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary with processed document
    """
    processor = PDFProcessor(document_path, chunk_size, chunk_overlap)
    return processor.process()
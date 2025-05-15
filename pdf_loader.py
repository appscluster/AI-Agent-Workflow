"""
PDF document processing module for extracting text, tables, and structural elements.
"""
import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.text_splitter import SentenceSplitter

class PDFProcessor:
    """
    Enhanced PDF processor that extracts text, tables, and structural elements
    from PDF documents while preserving document structure.
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        heading_split: bool = True,
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks to maintain context
            heading_split: Whether to split by headings for structure-aware chunking
        """        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_split = heading_split
        
        # Initialize the PyMuPDF reader
        self.reader = PyMuPDFReader()
        
        # Create a sentence splitter instead of token splitter for more natural document chunking
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # For the updated HierarchicalNodeParser API, we need a simpler approach
        # We'll use the sentence splitter directly
        self.node_parser = None  # Will handle chunking directly in process_document
    
    def load_document(self, file_path: str) -> Document:
        """
        Load a PDF document with basic text extraction.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document: LlamaIndex Document object
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Load document using PyMuPDFReader
        document = self.reader.load_data(file_path)
        print(f"Loaded document with {len(document)} pages")
        
        return document[0]  # Return the single document
    
    def process_document(self, file_path: str) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Process a PDF document with full extraction of text, tables, and structure.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple containing:
                - List of document chunks
                - Metadata dictionary with tables and structure information
        """
        # Load the document using PyMuPDF for advanced processing
        doc = fitz.open(file_path)
        
        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", Path(file_path).stem),
            "author": doc.metadata.get("author", "Unknown"),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "page_count": len(doc),
            "tables": [],
            "images": [],
            "toc": self._extract_toc(doc),
        }
        
        # Process each page to extract:
        # 1. Text with structure (headings, paragraphs)
        # 2. Tables (as dataframes and text)
        # 3. Image references and captions
        
        full_text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Extract tables from the page
            tables = self._extract_tables(page)
            for i, table in enumerate(tables):
                table_id = f"table_p{page_num + 1}_{i + 1}"
                metadata["tables"].append({
                    "id": table_id,
                    "page": page_num + 1,
                    "content": table.to_dict(),
                    "text_representation": table.to_string(index=False)
                })
            
            # Extract images and captions
            images = self._extract_images(page)
            metadata["images"].extend([{
                "id": f"img_p{page_num + 1}_{i + 1}",
                "page": page_num + 1,
                "caption": img_info.get("caption", ""),
                "is_chart": img_info.get("is_chart", False)
            } for i, img_info in enumerate(images)])        # Create LlamaIndex Document
        doc_obj = Document(text=full_text, metadata=metadata)
        
        # Split document into chunks using sentence splitter for more natural chunking
        # This can create more coherent chunks that align with sentence boundaries
        text_chunks = self.text_splitter.split_text(full_text)
        
        # Add structure information to metadata for each chunk
        # This helps maintain some context even with simple splitting
        # BUT we need to limit metadata size to avoid the "metadata too large" error
        chunk_nodes = []
        for i, chunk in enumerate(text_chunks):
            # Create a lightweight metadata copy with only essential information
            lightweight_metadata = {
                "title": metadata.get("title", ""),
                "chunk_id": i,
                "page_count": metadata.get("page_count", 0),
                "document_type": "pdf"
            }
            
            # Add TOC information but limit its size
            if "toc" in metadata and metadata["toc"]:
                # Only include first few TOC entries to limit size
                lightweight_metadata["toc"] = metadata["toc"][:5] if len(metadata["toc"]) > 5 else metadata["toc"]
            
            # Add references to tables and images but not their full content
            # This keeps the metadata references while reducing size
            if "tables" in metadata and metadata["tables"]:
                lightweight_metadata["table_refs"] = [
                    {"id": table["id"], "page": table["page"]} 
                    for table in metadata["tables"][:5]  # Further limit references to reduce size
                ]
            
            if "images" in metadata and metadata["images"]:
                lightweight_metadata["image_refs"] = [
                    {"id": img["id"], "page": img["page"], "is_chart": img["is_chart"]} 
                    for img in metadata["images"][:5]  # Further limit references to reduce size
                ]
            
            chunk_nodes.append(Document(text=chunk, metadata=lightweight_metadata))
        
        # Return both the chunked documents and the metadata
        return chunk_nodes, metadata
    
    def _extract_toc(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract table of contents (document structure).
        
        Args:
            doc: PyMuPDF document
            
        Returns:
            List of TOC entries with level, title and page number
        """
        toc = doc.get_toc()
        return [{"level": level, "title": title, "page": page} for level, title, page in toc]
    
    def _extract_tables(self, page: fitz.Page) -> List[pd.DataFrame]:
        """
        Extract tables from a page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of pandas DataFrames containing tables
        """
        tables = []
        
        try:
            # Get the page's text as a dict which preserves blocks
            blocks = page.get_text("dict")["blocks"]
            
            # Filter for blocks that look like tables (consecutive lines with similar x-coords)
            for block in blocks:
                if block["type"] == 0:  # Text block
                    lines = block.get("lines", [])
                    
                    # Simple heuristic to detect tables - aligned text spans
                    if len(lines) > 3:  # Minimum rows for a table
                        # Check if spans have consistent x-positions (column alignment)
                        x_positions = []
                        for line in lines:
                            spans = line.get("spans", [])
                            x_positions.append([span["bbox"][0] for span in spans])
                        
                        # If there's consistency in x-positions, treat as table
                        # This is a simple heuristic - could be improved
                        if self._check_column_alignment(x_positions) and len(x_positions[0]) > 1:
                            # Extract text from spans
                            table_data = []
                            for line in lines:
                                row = [span["text"] for span in line.get("spans", [])]
                                table_data.append(row)
                              # Ensure all rows have same number of columns
                            max_cols = max(len(row) for row in table_data)
                            for row in table_data:
                                while len(row) < max_cols:
                                    row.append("")
                            
                            # Create DataFrame with unique column names
                            if table_data:
                                # Create unique column headers to avoid warnings
                                if len(table_data) > 0:
                                    headers = table_data[0] if table_data else []
                                    # Make headers unique by adding numbers if duplicated
                                    unique_headers = []
                                    header_counts = {}
                                    
                                    for h in headers:
                                        if h in header_counts:
                                            header_counts[h] += 1
                                            unique_h = f"{h}_{header_counts[h]}"
                                        else:
                                            header_counts[h] = 0
                                            unique_h = h
                                        unique_headers.append(unique_h)
                                    
                                    df = pd.DataFrame(table_data[1:], columns=unique_headers)
                                else:
                                    # Generate default column names
                                    df = pd.DataFrame(table_data)
                                tables.append(df)
        
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        return tables
    
    def _check_column_alignment(self, x_positions: List[List[float]]) -> bool:
        """
        Check if x-positions show column alignment pattern.
        
        Args:
            x_positions: List of x-positions for each line
            
        Returns:
            Boolean indicating if positions suggest a table
        """
        # Simple heuristic - check if most rows have same number of columns
        # and if x-positions are relatively consistent
        if not x_positions:
            return False
            
        # Most common length
        lengths = [len(pos) for pos in x_positions]
        most_common_len = max(set(lengths), key=lengths.count)
        
        # If most rows have the same number of columns (>1), it might be a table
        if most_common_len <= 1:
            return False
            
        consistent_rows = sum(1 for l in lengths if l == most_common_len)
        consistency_ratio = consistent_rows / len(lengths)
        
        return consistency_ratio > 0.7  # At least 70% of rows have consistent columns
    
    def _extract_images(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """
        Extract images and their captions from a page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of dictionaries with image information
        """
        image_info = []
        
        try:
            # Extract image objects
            image_list = page.get_images(full=True)
            
            # Get the page's text blocks
            blocks = page.get_text("dict")["blocks"]
            
            for img_index, img in enumerate(image_list):
                # We would extract the actual image here if needed
                # img_xref = img[0]  # cross-reference number
                # base_image = doc.extract_image(img_xref)
                
                # Look for captions near the image
                caption = self._find_image_caption(blocks)
                
                # Determine if it's a chart (simple heuristic - look for keywords)
                is_chart = any(keyword in caption.lower() 
                               for keyword in ["chart", "figure", "graph", "plot", "diagram"])
                
                image_info.append({
                    "caption": caption,
                    "is_chart": is_chart
                })
        
        except Exception as e:
            print(f"Error extracting images: {e}")
        
        return image_info
    
    def _find_image_caption(self, blocks: List[Dict[str, Any]]) -> str:
        """
        Find potential image captions in text blocks.
        
        Args:
            blocks: Text blocks from a page
            
        Returns:
            Most likely caption text
        """
        # Simple heuristic - look for short text blocks starting with "Figure", "Chart", etc.
        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span["text"]
                
                # Check if this looks like a caption
                if len(block_text) < 200:  # Captions are usually short
                    lower_text = block_text.lower()
                    if (lower_text.startswith("figure") or 
                        lower_text.startswith("chart") or
                        lower_text.startswith("table") or
                        lower_text.startswith("graph") or
                        "fig." in lower_text):
                        return block_text
        
        return ""  # No caption found
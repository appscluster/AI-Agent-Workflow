# AI Agent Workflow for Document Question Answering

This project implements an AI-powered workflow capable of reading multi-modal business documents and answering complex natural language questions with context awareness and conversation memory.

## Overview

The system processes 20-25 page business PDFs containing mixed content (text, tables, charts) and uses an orchestrated graph-based workflow to answer questions accurately. The solution maintains context for follow-up queries and provides reasoned responses based on document content.

## Features

- PDF document parsing and chunking with structure preservation
- LlamaIndex-based orchestration for flexible reasoning
- Context-aware retrieval system
- Conversation memory for follow-up questions
- Modular architecture for easy maintenance and extension

## Architecture

The project follows a modular design with specialized components:

```
AI-Agent-Workflow/
├── main.py                # Main execution script
├── graph.py               # LlamaIndex Graph orchestration flow
├── pdf_loader.py          # PDF processing utilities
├── agents/
│   ├── planner.py         # Query planning agent
│   ├── retriever.py       # Context retrieval agent
│   └── answer_generator.py # Response composition agent
├── utils/
│   ├── memory.py          # Conversation state tracking
│   └── embeddings.py      # Vector embeddings for retrieval
├── answers.txt            # Sample responses
└── explanation.txt        # Implementation decisions
```

## Workflow

1. **Document Ingestion**: PDF is processed using PyMuPDF to extract text, tables, and structural elements
2. **Query Analysis**: Planner agent determines the type of question and required context
3. **Retrieval**: Relevant document chunks are fetched using embedding-based similarity
4. **Answer Generation**: Final response is composed with citations to source material
5. **Memory Management**: Conversation context is maintained for follow-up questions

## Requirements

- Python 3.10+
- Dependencies (see requirements.txt)
- LlamaIndex Graph orchestration library
- Vector database for document indexing

## Setup and Usage

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Process a document:
   ```
   python main.py --document path/to/document.pdf
   ```

3. Ask questions interactively:
   ```
   python main.py --document path/to/document.pdf --interactive
   ```

4. Process predefined questions:
   ```
   python main.py --document path/to/document.pdf --questions questions.txt
   ```

## Usage

Run the main script with a PDF document:

```bash
python main.py --document "path/to/document.pdf" --interactive
```

Process a document with predefined questions and save answers to a file:

```bash
python c:\home\AI-Agent-Workflow\main.py --document c:\home\AI-Agent-Workflow\samples\Test_Document.pdf --questions c:\home\AI-Agent-Workflow\test_questions.txt --output c:\home\AI-Agent-Workflow\temp_answers.txt
```

## Implementation Decisions

- **LlamaIndex** was chosen for orchestration due to its state management capabilities and flexible node structure
- **PyMuPDF** provides robust PDF parsing with layout awareness
- **Vector embeddings** enable semantic search for accurate information retrieval
- **Modular agents** allow for specialized reasoning tailored to different query types

## Limitations

- Limited image analysis capability for complex charts
- Processing time increases with document length
- Maximum document size limited to ~25 pages for efficient processing

## Future Improvements

- Enhanced chart and image analysis using OCR and image recognition
- Multi-document correlation for cross-referencing information
- Performance optimizations for larger documents
- Integration with real-time data sources

## Contributing

Welcome all contributions!

## Authors & Acknowledgment

Dr. Abdul Hamid - Project Owner - [LinkedIn](https://www.linkedin.com/in/ahceo/)

## License

[MIT License](LICENSE)

# AI Agent Workflow

## Installation

Before running this project, make sure to install the required dependencies:

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with specific versions
pip install -r requirements.txt
```

## Known Issues

If you encounter import errors related to pydantic and chromadb, ensure you're using the correct versions:
- pydantic < 2.0.0
- chromadb == 0.4.13

These specific versions are required for compatibility.


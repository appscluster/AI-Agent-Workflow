"""
Main execution script for document question answering workflow.
"""
import os
import argparse
import logging
from typing import List, Optional

from graph import QAWorkflowGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment variable.
    
    Returns:
        OpenAI API key if available, None otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    return api_key

def process_questions(
    workflow: QAWorkflowGraph,
    questions: List[str],
    output_file: Optional[str] = None
) -> List[dict]:
    """
    Process a list of questions and save answers.
    
    Args:
        workflow: QA workflow instance
        questions: List of questions
        output_file: Path to save answers (optional)
        
    Returns:
        List of question-answer pairs
    """
    results = []
    
    # Process each question
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        
        # Get answer from workflow
        result = workflow.answer_question(question)
        
        # Extract question and answer
        q = result["question"]
        a = result["answer"]
        
        # Print to console
        print(f"\nQ{i+1}: {q}")
        print(f"A{i+1}: {a}\n")
        
        # Add to results
        results.append({"question": q, "answer": a})
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            for i, qa in enumerate(results):
                f.write(f"Q{i+1}: {qa['question']}\n")
                f.write(f"A{i+1}: {qa['answer']}\n\n")
        
        logger.info(f"Answers saved to {output_file}")
    
    return results

def interactive_mode(workflow: QAWorkflowGraph):
    """
    Run in interactive mode, allowing user to ask questions.
    
    Args:
        workflow: QA workflow instance
    """
    print("\n=== Interactive Question Answering ===")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        # Get question from user
        question = input("Question: ").strip()
        
        # Check for exit command
        if question.lower() in ["exit", "quit"]:
            print("Exiting interactive mode")
            break
        
        # Skip empty questions
        if not question:
            continue
        
        # Get answer from workflow
        try:
            result = workflow.answer_question(question)
            
            # Extract answer
            answer = result["answer"]
            
            # Print answer
            print(f"\nAnswer: {answer}\n")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"An error occurred: {e}")

def process_questions_from_file(
    workflow: QAWorkflowGraph,
    questions_file: str,
    output_file: Optional[str] = None
) -> List[dict]:
    """
    Process questions from a file.
    
    Args:
        workflow: QA workflow instance
        questions_file: Path to file with questions
        output_file: Path to save answers (optional)
        
    Returns:
        List of question-answer pairs
    """
    # Read questions from file
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Process questions
    return process_questions(workflow, questions, output_file)

def main():
    """
    Main entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Document Question Answering System")
    parser.add_argument("--document", required=True, help="Path to the document PDF")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--questions", help="Path to file with questions")
    parser.add_argument("--output", default="answers.txt", help="Path to save answers")
    parser.add_argument("--persist-dir", help="Directory to persist embeddings and data")
    parser.add_argument("--no-kg", action="store_true", help="Disable knowledge graph")
    
    args = parser.parse_args()
    
    try:
        # Check if document exists
        if not os.path.exists(args.document):
            logger.error(f"Document not found: {args.document}")
            return
        
        # Get API key
        api_key = get_openai_api_key()
        
        # Initialize workflow
        logger.info("Initializing QA workflow")
        try:
            workflow = QAWorkflowGraph(
                document_path=args.document,
                persist_dir=args.persist_dir,
                use_knowledge_graph=not args.no_kg,
                openai_api_key=api_key
            )
        except ImportError as e:
            logger.warning(f"Couldn't initialize with knowledge graph: {e}")
            logger.warning("Falling back to basic implementation without knowledge graph")
            workflow = QAWorkflowGraph(
                document_path=args.document,
                persist_dir=args.persist_dir,
                use_knowledge_graph=False,
                openai_api_key=api_key
            )
        
        # Process document
        logger.info("Processing document")
        workflow.process_document()
        
        # Run in selected mode
        if args.interactive:
            # Interactive mode
            interactive_mode(workflow)
        elif args.questions:
            # Process questions from file
            process_questions_from_file(workflow, args.questions, args.output)
        else:
            # Default to sample questions
            sample_questions = [
                "What strategic goals were outlined for the next fiscal year?",
                "What risks were identified in the competitive landscape section?",
                "Summarize the key takeaways from the executive summary.",
                "Can you elaborate on the points you just mentioned?",
                "What were the financial projections related to these goals?"
            ]
            process_questions(workflow, sample_questions, args.output)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
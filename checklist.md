## AI Agent Planning & Execution Checklist

### 📄 Document Ingestion
- [x] Load PDF (20–25 pages) successfully
- [x] Extract clean text from body sections
- [x] Detect and extract tables if available
- [x] Extract image captions and chart references
- [x] Chunk text based on document structure (e.g., heading-level splitting)

### 🧠 Agent Architecture (Planning & Routing)
- [x] Choose orchestration framework (LangGraph / CrewAI / AutoGen / Custom)
- [x] Define node/agent roles:
  - [x] Ingestor
  - [x] Planner
  - [x] Retriever
  - [x] Answer Generator
  - [x] Memory Manager (for follow-ups)
- [x] Implement graph or flow logic for question routing
- [x] Define prompts and tools used by each node

### 🔍 Retrieval & Semantic Search
- [x] Create embeddings for document chunks (OpenAI / HuggingFace / local)
- [x] Store in a vector index (FAISS / Chroma / Qdrant)
- [x] Retrieve top-k relevant sections per question
- [x] (Optional) Rerank results for quality

### 🗣️ Response Generation
- [x] Generate coherent, grounded answer from retrieved context
- [x] Format output clearly (bullets, paragraphs, etc.)
- [x] Sanitize hallucinated content if detected

### 🔁 Follow-up Support
- [x] Maintain conversation memory across turns
- [x] Reference prior answers and refine based on context
- [x] Handle clarification or iterative questions

### 🧪 Sample Question Evaluation
- [x] Q1: "What strategic goals were outlined for the next fiscal year?" ✅
- [x] Q2: "What risks were identified in the competitive landscape section?" ✅
- [x] Q3: "Summarize the key takeaways from the executive summary." ✅
- [x] Q4–Q5: Follow-up contextual questions ✅

### 📦 Output & Submission
- [x] Save answers to `answers.txt`
- [x] Include architectural rationale in `explanation.txt`
- [x] Provide setup guide in `README.md`
- [x] Deliver full working code as a zipped project or GitHub repo

### 🧠 Bonus Features (Optional)
- [x] Add retry/reflection loop for low-confidence answers
- [x] Track graph state using LangGraph variables
- [x] Use OCR or layout parsers for image/chart analysis
- [x] Implement reranking of chunks using scoring heuristics

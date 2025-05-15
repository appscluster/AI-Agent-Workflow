## AI Agent Planning & Execution Checklist

### 📄 Document Ingestion
- [ ] Load PDF (20–25 pages) successfully
- [ ] Extract clean text from body sections
- [ ] Detect and extract tables if available
- [ ] Extract image captions and chart references
- [ ] Chunk text based on document structure (e.g., heading-level splitting)

### 🧠 Agent Architecture (Planning & Routing)
- [ ] Choose orchestration framework (LangGraph / CrewAI / AutoGen / Custom)
- [ ] Define node/agent roles:
  - [ ] Ingestor
  - [ ] Planner
  - [ ] Retriever
  - [ ] Answer Generator
  - [ ] Memory Manager (for follow-ups)
- [ ] Implement graph or flow logic for question routing
- [ ] Define prompts and tools used by each node

### 🔍 Retrieval & Semantic Search
- [ ] Create embeddings for document chunks (OpenAI / HuggingFace / local)
- [ ] Store in a vector index (FAISS / Chroma / Qdrant)
- [ ] Retrieve top-k relevant sections per question
- [ ] (Optional) Rerank results for quality

### 🗣️ Response Generation
- [ ] Generate coherent, grounded answer from retrieved context
- [ ] Format output clearly (bullets, paragraphs, etc.)
- [ ] Sanitize hallucinated content if detected

### 🔁 Follow-up Support
- [ ] Maintain conversation memory across turns
- [ ] Reference prior answers and refine based on context
- [ ] Handle clarification or iterative questions

### 🧪 Sample Question Evaluation
- [ ] Q1: "What strategic goals were outlined for the next fiscal year?" ✅
- [ ] Q2: "What risks were identified in the competitive landscape section?" ✅
- [ ] Q3: "Summarize the key takeaways from the executive summary." ✅
- [ ] Q4–Q5: Follow-up contextual questions ✅

### 📦 Output & Submission
- [ ] Save answers to `answers.txt`
- [ ] Include architectural rationale in `explanation.txt`
- [ ] Provide setup guide in `README.md`
- [ ] Deliver full working code as a zipped project or GitHub repo

### 🧠 Bonus Features (Optional)
- [ ] Add retry/reflection loop for low-confidence answers
- [ ] Track graph state using LangGraph variables
- [ ] Use OCR or layout parsers for image/chart analysis
- [ ] Implement reranking of chunks using scoring heuristics
